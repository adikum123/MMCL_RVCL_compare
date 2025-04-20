import argparse

import torch

from beta_crown.utils import print_args
from resnet.resnet_unsupervised_encoder import ResnetEncoder

parser = argparse.ArgumentParser(description="unsupervised verification")

# dataset args
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--dataset", default="cifar-10", type=str, help="cifar-10/mnist")
parser.add_argument(
    "--color_jitter_strength",
    default=0.5,
    type=float,
    help="0.5 for CIFAR, 1.0 for ImageNet",
)
# train args
parser.add_argument("--seed", default=1, type=int, help="random seed")
parser.add_argument(
    "--use_validation",
    action="store_true",
    help="Use validation set and stopping criteria",
)
parser.add_argument(
    "--adversarial",
    action="store_true",
    help="Use adversarial contrastive training",
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
parser.add_argument("--loss_type", type=str, default="", help=["Select mmcl, info_nce, nce, cosine or barlow"])
parser.add_argument(
    "--train_type",
    default="contrastive",
    type=str,
    help="contrastive/linear eval/test/supervised",
)
# args for mmcl loss
parser.add_argument("--kernel_gamma", type=str, default="auto")
parser.add_argument(
    "--svm_lr",
    default=1e-3,
    type=float,
    help="learning rate for SVM optimisation problem in MMCL",
)
parser.add_argument("--C", default=1.0, type=float, help="C for SVM")
parser.add_argument(
    "--deegre", default=3.0, type=float, help="Degree for polynomial kernel"
)
parser.add_argument("--kernel_type", default="rbf", type=str, help="Kernel Type")
parser.add_argument("--sigma", default=0.07, type=float, help="Sigma")
parser.add_argument(
    "--solver_type", default="nesterov", type=str, help="Type of PGD Solver"
)
parser.add_argument(
    "--use_norm", default="false", type=str, help="Use Norm - PGD Solver"
)

args = parser.parse_args()
print_args(args)
# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResnetEncoder(hparams=args, device=device)
model.train()
