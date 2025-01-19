import argparse

import torch

from beta_crown.utils import print_args
from mmcl.encoder import MMCL_Encoder
from mmcl.linear_eval import LinearEval

parser = argparse.ArgumentParser(description="unsupervised verification")

##### arguments for beta CROWN #####
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
    "--load_checkpoint", default="", type=str, help="PATH TO CHECKPOINT"
)
parser.add_argument("--seed", default=1, type=int, help="random seed")
# loss params
parser.add_argument("--C", default=1.0, type=float, help="C for SVM")
parser.add_argument(
    "--deegre", default=3.0, type=float, help="Degree for polynomial kernel"
)
parser.add_argument("--kernel_type", default="rbf", type=str, help="Kernel Type")
parser.add_argument("--sigma", default=0.07, type=float, help="Sigma")
parser.add_argument("--reg", default=0.1, type=float, help="Regularization")
parser.add_argument(
    "--use_validation",
    action="store_true",
    help="Use validation set and stopping criteria",
)
parser.add_argument(
    "--encoder_num_iters", default=500, type=int, help="Num iters - PGD Solver"
)
parser.add_argument(
    "--linear_eval_num_iters", default=100, type=int, help="Num iters - PGD Solver"
)
parser.add_argument("--eta", default=1e-5, type=float, help="Eta - PGD Solver")
parser.add_argument(
    "--stop_condition",
    default=1e-2,
    type=float,
    help="Stop Condition - PGD Solver",
)
parser.add_argument(
    "--solver_type", default="nesterov", type=str, help="Type of PGD Solver"
)
parser.add_argument(
    "--use_norm", default="false", type=str, help="Use Norm - PGD Solver"
)

# data params
parser.add_argument("--multiplier", default=2, type=int)
parser.add_argument(
    "--dist",
    default="dp",
    type=str,
    help="dp: DataParallel, ddp: DistributedDataParallel",
    choices=["dp", "ddp"],
)
parser.add_argument(
    "--color_dist_s", default=1.0, type=float, help="Color distortion strength"
)
parser.add_argument(
    "--scale_lower",
    default=0.08,
    type=float,
    help="The minimum scale factor for RandomResizedCrop",
)
parser.add_argument(
    "--sync_bn",
    default=True,
    type=bool,
    help="Syncronises BatchNorm layers between all processes if True",
)
parser.add_argument(
    "--color_jitter_strength",
    default=0.5,
    type=float,
    help="0.5 for CIFAR, 1.0 for ImageNet",
)
parser.add_argument(
    "--encoder_lr", default=1e-3, type=float, help="learning rate for MMCL encoder"
)
parser.add_argument(
    "--svm_lr",
    default=1e-3,
    type=float,
    help="learning rate for SVM optimisation problem in MMCL",
)
parser.add_argument(
    "--linear_eval_lr",
    default=1e-3,
    type=float,
    help="learning rate for linear eval on top of MMCL encoder",
)
parser.add_argument("--step_size", default=30, type=int, help="scheduler step size")
parser.add_argument(
    "--criterion_to_use",
    default="mmcl_pgd",
    type=str,
    help="choose which mmcl svm solver to use",
)
parser.add_argument("--kernel_gamma", type=str, default="auto")
parser.add_argument("--scheduler_gamma", type=float, default=0.1)

##### arguments for RoCL Linear eval #####
parser.add_argument("--trans", action="store_true", help="use transformed sample")
parser.add_argument("--clean", type=bool, default=True, help="use clean sample")
parser.add_argument("--adv_img", action="store_true", help="use adversarial sample")
parser.add_argument("--finetune", action="store_true", help="finetune the model")
parser.add_argument(
    "--ss", action="store_true", help="using self-supervised learning loss"
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
# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")
model = MMCL_Encoder(hparams=args, device=device)
model.train()
model.save()

# Test model
args.train_type = "linear_eval"
linear_eval = LinearEval(hparams=args, device=device, encoder=model)
linear_eval.train()
linear_eval.test()
