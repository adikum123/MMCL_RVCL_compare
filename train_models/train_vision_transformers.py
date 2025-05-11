import argparse

import torch

from beta_crown.utils import print_args
from vision_transformers.vision_transformers_model import \
    VisionTransformerModel

parser = argparse.ArgumentParser(description="Resnet unsupervised classifier")

parser.add_argument("--batch_size", type=int, default=256, help="batch size")
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
    "--lr", default=1e-4, type=float, help="Learning rate"
)
parser.add_argument("--step_size", default=30, type=int, help="scheduler step size")
parser.add_argument("--scheduler_gamma", type=float, default=0.1)

args = parser.parse_args()
print_args(args)
# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")
model = VisionTransformerModel(hparams=args, device=device)
model.train()
model.save()
model.test()
