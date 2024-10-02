import torch.nn as nn
import torch
import torch.nn.functional as F
   
import math
class PALM(nn.Module):
    def __init__(self, args, num_classes=10, n_protos=1000, proto_m=0.99, temp=0.1, lambda_pcon=1, k=0,feat_dim=128, epsilon=0.05):
        super(PALM, self).__init__()
        self.num_classes = num_classes
        self.temp = temp  # temperature scaling
        self.nviews = args.nviews
        self.cache_size = args.cache_size
        
        self.lambda_pcon = lambda_pcon
        
        self.feat_dim = feat_dim
        
        self.epsilon = epsilon
        self.sinkhorn_iterations = 3
        self.k = min(k, self.cache_size)
        
        self.n_protos = n_protos
        self.proto_m = proto_m
        self.register_buffer("protos", torch.rand(self.n_protos,feat_dim))
        self.protos = F.normalize(self.protos, dim=-1)
        
    def sinkhorn(self, features):
        out = torch.matmul(features, self.protos.detach().T)
            
        Q = torch.exp(out.detach() / self.epsilon).t()#
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if torch.isinf(sum_Q):
            self.protos = F.normalize(self.protos, dim=1, p=2)
            out = torch.matmul(features, self.protos.detach().T)
            Q = torch.exp(out.detach() / self.epsilon).t()#
            sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            Q = F.normalize(Q, dim=1, p=1)
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q = F.normalize(Q, dim=0, p=1)
            Q /= B

        Q *= B
        return Q.t()
        
    def mle_loss(self, features, targets):
        # update prototypes by EMA

        features = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_labels = targets.contiguous().repeat(self.nviews).view(-1, 1)
        contrast_labels = torch.arange(self.num_classes).repeat(self.cache_size).view(-1,1).cuda()
        mask = torch.eq(anchor_labels, contrast_labels.T).float().cuda()
                
        Q = self.sinkhorn(features)
        # topk
        if self.k > 0:
            update_mask = mask*Q
            _, topk_idx = torch.topk(update_mask, self.k, dim=1)
            topk_mask = torch.scatter(
                torch.zeros_like(update_mask),
                1,
                topk_idx,
                1
            ).cuda()
            update_mask = F.normalize(F.normalize(topk_mask*update_mask, dim=1, p=1),dim=0, p=1)
        # original
        else:
            update_mask = F.normalize(F.normalize(mask * Q, dim=1, p=1),dim=0, p=1)
        update_features = torch.matmul(update_mask.T, features)
        
        protos = self.protos
        protos = self.proto_m * protos + (1-self.proto_m) * update_features

        self.protos = F.normalize(protos, dim=1, p=2)
        
        Q = self.sinkhorn(features)
        
        proto_dis = torch.matmul(features, self.protos.detach().T)
        anchor_dot_contrast = torch.div(proto_dis, self.temp)
        logits = anchor_dot_contrast
       
        if self.k > 0:
            loss_mask = mask*Q
            _, topk_idx = torch.topk(update_mask, self.k, dim=1)
            topk_mask = torch.scatter(
                torch.zeros_like(update_mask),
                1,
                topk_idx,
                1
            ).cuda()
            loss_mask = F.normalize(topk_mask*loss_mask, dim=1, p=1)
            masked_logits = loss_mask * logits 
        else:  
            masked_logits = F.normalize(Q*mask, dim=1, p=1) * logits
    
        pos=torch.sum(masked_logits, dim=1)
        neg=torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))
        log_prob=pos-neg
        
        loss = -torch.mean(log_prob)
        return loss   
    
    def proto_contra(self):
        
        protos = F.normalize(self.protos, dim=1)
        batch_size = self.num_classes
        
        proto_labels = torch.arange(self.num_classes).repeat(self.cache_size).view(-1,1).cuda()
        mask = torch.eq(proto_labels, proto_labels.T).float().cuda()    

        contrast_count = self.cache_size
        contrast_feature = protos

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            0.5)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to('cuda'),
            0
        )
        mask = mask*logits_mask
        
        pos = torch.sum(F.normalize(mask, dim=1, p=1)*logits, dim=1)
        neg=torch.log(torch.sum(logits_mask * torch.exp(logits), dim=1))
        log_prob=pos-neg

        # loss
        loss = - torch.mean(log_prob)
        return loss
    
           
    def forward(self, features, targets):
        loss = 0
        loss_dict = {}

        g_con = self.mle_loss(features, targets)

        loss += g_con
        loss_dict['mle'] = g_con.cpu().item()

        if self.lambda_pcon > 0:            
            g_dis = self.lambda_pcon * self.proto_contra()
            loss += g_dis
            loss_dict['proto_contra'] = g_dis.cpu().item()
                                
        self.protos = self.protos.detach()

        return loss, loss_dict


import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
class PLAG(nn.Module):
    def __init__(self, args, num_classes=10, n_protos=1000, proto_m=0.99, temp=0.1, lambda_pcon=1, k=0, feat_dim=64,
                 epsilon=1.0):
        super(PLAG, self).__init__()
        self.num_classes = num_classes
        self.temp = temp  # temperature scaling
        self.nviews = args.nviews
        self.cache_size = args.cache_size

        self.lambda_pcon = lambda_pcon

        self.feat_dim = feat_dim
        self.args = args
        self.epsilon = epsilon
        self.sinkhorn_iterations = 3
        self.k = min(k, self.cache_size)
        self.inv_w = args.inv_w
        self.n_protos = n_protos
        self.proto_m = proto_m
        self.register_buffer("protos", torch.randn(self.n_protos, feat_dim))
        self.protos.requires_grad = True
        self.protos = F.normalize(self.protos, dim=-1)
        self.att_weight_q = nn.Linear(feat_dim , args.att_dim)
        self.att_weight_k = nn.Linear(feat_dim, args.att_dim)
        self.att_weight_v = nn.Linear(feat_dim, feat_dim)
        self.norm_fact = 1 / math.sqrt(args.att_dim)
        self.mix_proj = nn.Sequential(nn.Linear(feat_dim * 2, feat_dim),
                                      nn.BatchNorm1d(feat_dim),
                                      nn.ReLU(),
                                      nn.Dropout(),
                                      nn.Linear(feat_dim, feat_dim))
    def sinkhorn(self, features):

        out = torch.matmul(features, self.protos.detach().T.cuda())

        Q = torch.exp(out.detach() / self.epsilon).t()  #
        B = Q.shape[1]
        K = Q.shape[0]


        sum_Q = torch.sum(Q)
        while torch.isinf(sum_Q):
            self.protos = F.normalize(self.protos, dim=1, p=2)
            out = torch.matmul(features, self.protos.detach().T)
            Q = torch.exp(out.detach() / self.epsilon).t()
            sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(self.sinkhorn_iterations):

            Q = F.normalize(Q, dim=1, p=1)
            Q /= K

            Q = F.normalize(Q, dim=0, p=1)
            Q /= B

        Q *= B
        return Q.t()
    def self_attention(self, features):
        query = self.att_weight_q(features)
        key = self.att_weight_k(self.protos)
        attention_scores = torch.matmul(query, key.T) * self.norm_fact

        attention_weights = F.softmax(attention_scores)
        return attention_weights
    def infer(self,features):
        if self.num_classes ==1 :
            proto_dis = torch.matmul(features, self.protos.detach().T)
            Q = self.self_attention(features)
            proto_dis = proto_dis * Q
            median_values, _ = torch.median(proto_dis, dim=1)
            pred = 1 - median_values.unsqueeze(1)
        else:
            proto_dis = torch.matmul(features, self.protos.detach().T)
            Q = self.self_attention(features)
            proto_dis = proto_dis*Q
            proto_reshaped = proto_dis.view(features.shape[0], self.num_classes, self.cache_size)

            pred, _ = torch.max(proto_reshaped, dim=2)

        return pred
    def mle_loss(self, features, targets):

        device = features.device
        if self.num_classes == 1 :

            targets = torch.squeeze(targets)
            anchor_labels = targets.contiguous().view(-1, 1)
            self.contrast_labels = torch.arange(self.num_classes).repeat(self.cache_size).view(-1, 1).to(device)

            mask = torch.eq(anchor_labels, self.contrast_labels.T).float().to(device)

            Q = self.self_attention(features)

            if self.k > 0:
                update_mask = mask * Q
                topk_value, topk_idx = torch.topk(update_mask, self.k, dim=1)
                topk_mask = torch.scatter(
                    torch.zeros_like(update_mask),
                    1,
                    topk_idx,
                    topk_value
                ).to(device)
                update_mask = topk_mask
            else:
                update_mask = F.normalize(F.normalize(mask * Q, dim=1, p=1), dim=0, p=1)
            update_features = torch.matmul(update_mask.T, features)

            protos = self.protos.to(device)
            up_prototye =  (1 - self.proto_m) * update_features.to(device)
            protos = self.proto_m * protos + up_prototye.to(device)

            self.protos = F.normalize(protos, dim=1, p=2)


            proto_dis = torch.matmul(features, self.protos.detach().T)
            median_values, _ = torch.median(proto_dis, dim=1)
            pred = 1-median_values.unsqueeze(1)
        else:

            targets = torch.squeeze(targets)
            anchor_labels = targets.contiguous().view(-1, 1)
            self.contrast_labels = torch.arange(self.num_classes).repeat(self.cache_size).view(-1, 1).to(device)

            mask = torch.eq(anchor_labels, self.contrast_labels.T).float().to(device)

            Q = self.self_attention(features)

            if self.k > 0:
                update_mask = mask * Q
                topk_value, topk_idx = torch.topk(update_mask, self.k, dim=1)
                topk_mask = torch.scatter(
                    torch.zeros_like(update_mask),
                    1,
                    topk_idx,
                    topk_value
                ).to(device)
                update_mask = topk_mask
            else:
                update_mask = F.normalize(F.normalize(mask * Q, dim=1, p=1), dim=0, p=1)
            update_features = torch.matmul(update_mask.T, features)

            protos = self.protos.to(device)
            up_prototye = (1 - self.proto_m) * update_features.to(device)
            protos = self.proto_m * protos + up_prototye.to(device)

            self.protos = F.normalize(protos, dim=1, p=2)

            proto_dis = torch.matmul(features, self.protos.detach().T)

            Q = self.self_attention(features)
            proto_dis = (proto_dis)*Q

            proto_reshaped = proto_dis.view(features.shape[0], self.num_classes, self.cache_size)

            pred, _ = torch.max(proto_reshaped, dim=2)


        return pred


    def proto_contra(self):
        protos = F.normalize(self.protos, dim=1)
        batch_size = self.num_classes

        proto_labels = torch.arange(self.num_classes).repeat(self.cache_size).view(-1, 1).cuda()
        mask = torch.eq(proto_labels, proto_labels.T).float().cuda()

        contrast_count = self.cache_size
        contrast_feature = protos

        anchor_feature = contrast_feature
        anchor_count = contrast_count


        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            0.5)

        logits = anchor_dot_contrast


        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to('cuda'),
            0
        )
        mask = mask * logits_mask

        pos = torch.sum(mask * logits, dim=1)
        neg = torch.log(torch.sum(logits_mask * torch.exp(logits), dim=1))
        log_prob = pos - neg
        loss = - torch.mean(log_prob)
        return loss
    def mix_cs_proj(self, c_f: torch.Tensor, s_f: torch.Tensor):
        n = c_f.size(0)
        perm = np.random.permutation(n)
        mix_f = torch.cat([c_f, s_f[perm]], dim=-1)
        proj_mix_f = self.mix_proj(mix_f)
        return proj_mix_f

    def simsiam_loss(self, mix_rep, targets):

        targets = torch.squeeze(targets)
        anchor_labels = targets.contiguous().view(-1, 1)
        device = targets.device

        mask = torch.eq(anchor_labels, self.contrast_labels.T).float().to(device)
        proto_dis = torch.matmul(mix_rep, self.protos.detach().T)
        proto_dis = F.sigmoid(mask*proto_dis)
        return -proto_dis.sum(dim=1).mean()


    def causal_loss(self,batch_tensor, pred,labels):

        same_label_mask = labels == labels.T
        diff_label_mask = labels != labels.T

        diff_pred_mask = pred != labels

        all = torch.matmul(batch_tensor, batch_tensor.T)
        neg = all[diff_pred_mask].sum()

        loss_p = all[same_label_mask].sum()/(neg+all[same_label_mask].sum())
        return -loss_p
    def forward(self, features, s_features, c_rep,s_feat,targets, loss_func, mask):
        loss = 0
        loss_dict = {}

        g_con = self.mle_loss(features, targets)

        cls_loss = loss_func(g_con, targets.float(), reduction='none') * mask

        cls_loss = cls_loss.sum() / mask.sum()

        loss += cls_loss


        loss_dict['mle'] = cls_loss.cpu().item()

        if self.lambda_pcon > 0:
            g_dis = self.lambda_pcon * self.proto_contra()
            loss += g_dis
            loss_dict['proto_contra'] = g_dis.cpu().item()

        if self.inv_w>0 :
            pred = torch.argmax(self.infer(features),dim=1)
            inv_loss = self.inv_w*self.causal_loss(features, pred, targets)
            loss_dict['inv_loss'] = inv_loss.cpu().item()
        self.protos = self.protos.detach()


        return loss, loss_dict


