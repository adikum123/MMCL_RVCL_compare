import argparse

import torch
from encoder import MMCL_Encoder
from linear_eval import LinearEval

parser = argparse.ArgumentParser(description="unsupervised verification")

##### arguments for beta CROWN #####
parser.add_argument(
    "--model",
    type=str,
    default="cnn_4layer_b",
    help="model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)",
)
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
    "--kernel_type", default="rbf", type=str, help="Kernel Type"
)
parser.add_argument("--sigma", default=0.07, type=float, help="Sigma")
parser.add_argument("--reg", default=0.1, type=float, help="Regularization")

parser.add_argument(
    "--num_iters", default=100, type=int, help="Num iters - PGD Solver"
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
    "--use_norm", default="true", type=str, help="Use Norm - PGD Solver"
)

# data params
parser.add_argument("--multiplier", default=2, type=int)
parser.add_argument('--dist', default='dp', type=str,
    help='dp: DataParallel, ddp: DistributedDataParallel',
    choices=['dp', 'ddp'],
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
# ddp
parser.add_argument(
    "--sync_bn",
    default=True,
    type=bool,
    help="Syncronises BatchNorm layers between all processes if True",
)
parser.add_argument('--color_jitter_strength', default=0.5, type=float, help='0.5 for CIFAR, 1.0 for ImageNet')
parser.add_argument('--encoder_lr', default=1e-5, type=float, help='learning rate for MMCL encoder')
parser.add_argument('--svm_lr', default=1e-5, type=float, help='learning rate for SVM optimisation problem in MMCL')
parser.add_argument('--linear_eval_lr', default=1e-5, type=float, help='learning rate for linear eval on top of MMCL encoder')
parser.add_argument('--step_size', default=10, type=int, help='scheduler step size')
parser.add_argument('--criterion_to_use', default='mmcl_pgd', type=str, help='choose which mmcl svm solver to use')
parser.add_argument('--gamma', type=str, default="auto")
parser.add_argument('--device', type=str, default='gpu')
args = parser.parse_args()

# Train model
print(f'Running on: {args.device}')
model = MMCL_Encoder(hparams=args, device=torch.device('cuda' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu'))
model.train()

# Test model
model.eval()
linear_eval = LinearEval(hparams=args, encoder=model)
linear_eval.train()
linear_eval.test()