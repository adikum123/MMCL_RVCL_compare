import argparse

import torch

from beta_crown.utils import print_args
from resnet.resnet_unsupervised import ResnetUnsupervised

parser = argparse.ArgumentParser(description="unsupervised verification")

parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--dataset", default="cifar-10", type=str, help="cifar-10/mnist")
parser.add_argument("--seed", default=1, type=int, help="random seed")
parser.add_argument(
    "--use_validation",
    action="store_true",
    help="Use validation set and stopping criteria",
)
parser.add_argument(
    "--num_iters", default=100, type=int, help="Num iters - PGD Solver"
)
parser.add_argument(
    "--lr",
    default=1e-3,
    type=float,
    help="learning rate for linear eval on top of MMCL encoder",
)
parser.add_argument("--step_size", default=50, type=int, help="scheduler step size")
parser.add_argument("--scheduler_gamma", type=float, default=0.1)
parser.add_argument("--resnet_unsupervised_ckpt", default="", type=str, help="Path to the checkpoint to reset the supervised model")
parser.add_argument("--relu_layer", action="store_true", help="Add relu layer")
parser.add_argument("--finetune", action="store_true", help="Store the finetune encoder")
parser.add_argument("--finetune_num_layers", type=int, help="Number of layers to finetune")
args = parser.parse_args()
print_args(args)
# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResnetUnsupervised(hparams=args, device=device)
model.train()
model.test()
