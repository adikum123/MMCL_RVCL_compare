import argparse

import torch

from beta_crown.utils import print_args
from regular_cl import RegularCLModel

parser = argparse.ArgumentParser(description="unsupervised verification")

##### arguments for beta CROWN #####
parser.add_argument(
    "--model_save_name",
    type=str,
    default=None,
    help="Name under which model will be saved in mmcl/encoder",
)
parser.add_argument(
    "--model",
    type=str,
    default="cnn_4layer_b",
    help="model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)",
)
parser.add_argument("--name", default="", type=str, help="name of run")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
##### arguments for model #####
parser.add_argument(
    "--train_type",
    default="contrastive",
    type=str,
    help="contrastive/linear eval/test/supervised",
)
parser.add_argument("--dataset", default="cifar-10", type=str, help="cifar-10/mnist")
parser.add_argument(
    "--regular_cl_checkpoint", default="", type=str, help="PATH TO CHECKPOINT"
)
parser.add_argument("--seed", default=1, type=int, help="random seed")
parser.add_argument("--num_iters", default=500, type=int, help="Num iters for training")
# data params
parser.add_argument("--multiplier", default=2, type=int)
parser.add_argument(
    "--color_jitter_strength",
    default=0.5,
    type=float,
    help="0.5 for CIFAR, 1.0 for ImageNet",
)
parser.add_argument(
    "--lr", default=1e-3, type=float, help="learning rate for MMCL encoder"
)
parser.add_argument("--step_size", default=30, type=int, help="scheduler step size")
parser.add_argument("--scheduler_gamma", default=0.1, type=float, help="gamma value for scheduler")

args = parser.parse_args()
print("Args:\n")
print_args(args)
# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")
model = RegularCLModel(hparams=args, device=device)
model.train()
model.save()
