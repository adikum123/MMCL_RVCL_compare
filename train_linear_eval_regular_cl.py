import argparse

import torch

from beta_crown.utils import print_args
from mmcl.linear_eval import LinearEval
from regular_cl import RegularCLModel

parser = argparse.ArgumentParser(description="unsupervised verification")

parser.add_argument("--name", default="", type=str, help="name of run")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
##### arguments for model #####
parser.add_argument(
    "--train_type",
    default="linear_eval",
    type=str,
    help="contrastive/linear eval/test/supervised",
)
parser.add_argument("--dataset", default="cifar-10", type=str, help="cifar-10/mnist")

parser.add_argument("--seed", default=1, type=int, help="random seed")
parser.add_argument("--reg", default=0.1, type=float, help="Regularization")
parser.add_argument(
    "--use_validation",
    action="store_true",
    help="Use validation set and stopping criteria",
)
parser.add_argument(
    "--num_iters", default=100, type=int, help="Num iters - PGD Solver"
)

# data params
parser.add_argument(
    "--color_jitter_strength",
    default=0.5,
    type=float,
    help="0.5 for CIFAR, 1.0 for ImageNet",
)
parser.add_argument(
    "--lr",
    default=1e-3,
    type=float,
    help="learning rate for linear eval on top of MMCL encoder",
)
parser.add_argument(
    "--model",
    type=str,
    default="cnn_4layer_b",
    help="model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)",
)
parser.add_argument("--step_size", default=30, type=int, help="scheduler step size")
parser.add_argument("--scheduler_gamma", type=float, default=0.1)
##### arguments for RoCL Linear eval #####
parser.add_argument("--trans", action="store_true", help="use transformed sample")
parser.add_argument("--clean", action="store_true", help="use clean sample")
parser.add_argument("--adv_img", action="store_true", help="use adversarial sample")
parser.add_argument("--finetune", action="store_true", help="finetune the model")
parser.add_argument(
    "--ss", action="store_true", help="using self-supervised learning loss"
)
parser.add_argument(
    "--regular_cl_checkpoint", default="", type=str
)
##### arguments for PGD attack & Adversarial Training #####
parser.add_argument("--attack_type", type=str, default="linf", help="adversarial l_p")
parser.add_argument(
    "--epsilon",
    type=float,
    default=8.0 / 255,
    help="maximum perturbation of adversaries (8/255(0.0314) for cifar-10)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.007,
    help="movement multiplier per iteration when generating adversarial examples (2/255=0.00784)",
)
parser.add_argument(
    "--k",
    type=int,
    default=10,
    help="maximum iteration when generating adversarial examples",
)
parser.add_argument("--random_start", type=bool, default=True, help="True for PGD")


args = parser.parse_args()
print_args(args)
assert args.regular_cl_load_checkpoint != "", f"Empty load checkpoint provided: {args.regular_cl_load_checkpoint}"
# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")
encoder = MMCL_Encoder(hparams=args, device=device)
model = LinearEval(hparams=args, encoder=encoder, device=device)
model.train()
model.save()
model.test()
