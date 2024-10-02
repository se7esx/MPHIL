import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # exp
    parser.add_argument("--exp_name", default="run", type=str,
                        help="Experiment name")
    parser.add_argument("--model_name", default="palg", type=str,
                        help="Model name")
    parser.add_argument("--dump_path", default="dump/", type=str,
                        help="Experiment dump path")
    parser.add_argument("--exp_id", default="", type=str,
                        help="Experiment ID")
    parser.add_argument("--gpu_id", default='0', type=str)
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--load_path", default=None, type=str)

    # dataset
    parser.add_argument("--data_root", default='data', type=str)
    parser.add_argument("--config_path", default='configs', type=str)
    parser.add_argument("--dataset", default='GOODHIV', type=str)
    parser.add_argument("--domain", default='scaffold', type=str)
    parser.add_argument("--shift", default='covariate', type=str)

    # VQ
    parser.add_argument("--num_e", default=4000, type=int)
    parser.add_argument("--commitment_weight", default=0.1, type=float)

    # Encoder
    parser.add_argument("--emb_dim", default=128, type=int)
    parser.add_argument("--layer", default=4, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--gnn_type", default='gin', type=str, choices=['gcn', 'gin'])
    parser.add_argument("--pooling_type", default='mean', type=str)

    # Model
    parser.add_argument("--inv_w", default=0.01, type=float)
    parser.add_argument("--reg_w", default=0.5, type=float)
    parser.add_argument("--gamma", default=0.9, type=float)

    # Training
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--bs", default=128, type=int)
    parser.add_argument("--epoch", default=200, type=int)

    #prototype
    # prototypes arguments
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, help="SGD momentum")
    parser.add_argument('--proto_m', default=0.9, type=float, help="prototypes update momentum")
    parser.add_argument('--cache-size', default=3, type=int)
    parser.add_argument('--nviews', default=2, type=int)
    parser.add_argument('--channels', default=300, type=int)
    parser.add_argument('--r', default=0.8, type=float, help='causal_ratio')
    parser.add_argument('--feat_dim', default=300, type=int)
    parser.add_argument('--att_dim', default=128, type=int)
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')
    # loss config
    parser.add_argument('--lambda_pcon', default=1, type=float)
    parser.add_argument('--epsilon', default=1, type=float)
    args = parser.parse_args()

    return args
