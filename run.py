import os
import logging
from tqdm import tqdm
from munch import Munch, munchify

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import numpy as np

from GOOD import register
from GOOD.utils.config_reader import load_config
from GOOD.utils.metric import Metric
from GOOD.data.dataset_manager import read_meta_info
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from GOOD.utils.train import nan2zero_get_mask

from args_parse import args_parser
from exputils import initialize_exp, set_seed, get_dump_path, describe_model, save_model, load_model
from models import MyModel, MyModel_p
from dataset import DrugOODDataset
from loss_functions import *

logger = logging.getLogger()

import warnings

warnings.filterwarnings("ignore")


def set_requires_grad(nets, requires_grad=False):

    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def initialize_weights(layer):
    if isinstance(layer, nn.Linear):

        nn.init.xavier_uniform_(layer.weight)

        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class Runner:
    def __init__(self, args, writer, logger_path):
        self.args = args
        self.device = torch.device('cuda', int(args.gpu_id))

        if args.dataset.startswith('GOOD'):
            # for GOOD, load Config
            cfg_path = os.path.join(args.config_path, args.dataset, args.domain, args.shift, 'base.yaml')
            cfg, _, _ = load_config(path=cfg_path)
            cfg = munchify(cfg)
            cfg.device = self.device
            dataset, meta_info = register.datasets[cfg.dataset.dataset_name].load(dataset_root=args.data_root,
                                                                                  domain=cfg.dataset.domain,
                                                                                  shift=cfg.dataset.shift_type,
                                                                                  generate=cfg.dataset.generate)
            
            read_meta_info(meta_info, cfg)

            cfg.model.dropout_rate = args.dropout
            cfg.train.train_bs = args.bs
            cfg.random_seed = args.random_seed

            loader = register.dataloader[cfg.dataset.dataloader_name].setup(dataset, cfg)
            self.train_loader = loader['train']
            self.valid_loader = loader['val']
            self.test_loader = loader['test']

            self.metric = Metric()
            self.metric.set_score_func(dataset['metric'] if type(dataset) is dict else getattr(dataset, 'metric'))
            self.metric.set_loss_func(dataset['task'] if type(dataset) is dict else getattr(dataset, 'task'))
            cfg.metric = self.metric
            self.num_classes = cfg.dataset.num_classes
        else:
            # DrugOOD
            dataset = DrugOODDataset(name=args.dataset, root=args.data_root)
            self.train_set = dataset[dataset.train_index]
            self.valid_set = dataset[dataset.valid_index]
            self.test_set = dataset[dataset.test_index]

            self.train_loader = DataLoader(self.train_set, batch_size=args.bs, shuffle=True, drop_last=True)
            self.valid_loader = DataLoader(self.valid_set, batch_size=args.bs, shuffle=False)
            self.test_loader = DataLoader(self.test_set, batch_size=args.bs, shuffle=False)
            self.metric = Metric()
            self.metric.set_loss_func(task_name='Binary classification')
            self.metric.set_score_func(metric_name='ROC-AUC')
            cfg = Munch()
            cfg.metric = self.metric
            cfg.model = Munch()
            cfg.model.model_level = 'graph'

            self.num_classes = 1

        if args.model_name == 'palg':
            self.model = MyModel_p(args=args, config=cfg).to(self.device)
            self.criterion = PLAG(args, temp=args.temp, num_classes=self.num_classes, proto_m=args.proto_m,
                                  n_protos=self.num_classes * args.cache_size, k=args.k, feat_dim=args.feat_dim,
                                  lambda_pcon=args.lambda_pcon).to(self.device)

            self.opt = torch.optim.Adam(list(self.model.parameters()) + list(self.criterion.parameters()), lr=args.lr)

        else:
            self.model = MyModel(args=args, config=cfg).to(self.device)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt,
                                                              milestones=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                                              gamma=0.95)
        self.total_step = 0
        self.writer = writer
        describe_model(self.model, path=logger_path)
        self.logger_path = logger_path

        self.cfg = cfg

    def run(self):
        if self.metric.lower_better == 1:
            best_valid_score, best_test_score = float('inf'), float('inf')
        else:
            best_valid_score, best_test_score = -1, -1

        for e in range(self.args.epoch):
            self.train_step(e)
            valid_score = self.test_step(self.valid_loader)

            logger.info(f"E={e}, valid={valid_score:.5f}, test-score={best_test_score:.5f}")

            if (valid_score > best_valid_score and self.metric.lower_better == -1) or \
                    (valid_score < best_valid_score and self.metric.lower_better == 1):
                test_score = self.test_step(self.test_loader)
                best_valid_score = valid_score
                best_test_score = test_score
                logger.info(f"UPDATE test-score={best_test_score:.5f}")

        logger.info(f"test-score={best_test_score:.5f}")

    def train_step(self, epoch):
        self.model.train()


        pbar = tqdm(self.train_loader, desc=f"E [{epoch}]")

        for data in pbar:
            data = data.to(self.device)
            causal_rep, s_graph_feat, c_rep, s_feat, loss_reg = self.model(data)
            mask, target = nan2zero_get_mask(data, 'None', self.cfg)

            loss, l_dict = self.criterion(causal_rep, s_graph_feat, c_rep, s_feat, target, self.metric.loss_func, mask)

            loss += self.args.reg_w * loss_reg

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.opt.step()

            pbar.set_postfix_str(f"loss={loss.item():.4f}")
            self.writer.add_scalar('loss', loss.item(), self.total_step)

            self.total_step += 1

    @torch.no_grad()
    def test_step(self, loader):

        self.model.eval()
        y_pred, y_gt = [], []
        for data in loader:
            data = data.to(self.device)

            if self.args.model_name == 'palg':
                causal_rep, s_graph_feat, c_rep, s_feat, loss_reg = self.model(data)
                pred = self.criterion.infer(causal_rep)
            else:
                pred, c_graph_feat, s_graph_feat, cmt_loss, loss_reg = self.model(data)

            mask, _ = nan2zero_get_mask(data, 'None', self.cfg)
            pred, target = eval_data_preprocess(data.y, pred, mask, self.cfg)
            y_pred.append(pred)
            y_gt.append(target)

        score = eval_score(y_pred, y_gt, self.cfg)

        return score

    def simsiam_loss(self, causal_rep, mix_rep):
        causal_rep = causal_rep.detach()
        causal_rep = F.normalize(causal_rep, dim=1)
        mix_rep = F.normalize(mix_rep, dim=1)
        return -(causal_rep * mix_rep).sum(dim=1).mean()


def main():
    args = args_parser()
    torch.cuda.set_device(int(args.gpu_id))

    logger = initialize_exp(args)
    set_seed(args.random_seed)
    logger_path = get_dump_path(args)
    writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard'))

    runner = Runner(args, writer, logger_path)
    runner.run()
    writer.close()


if __name__ == '__main__':
    main()
