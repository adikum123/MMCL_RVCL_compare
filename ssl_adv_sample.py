import argparse

import torch

import rocl.data_loader as data_loader
from beta_crown.utils import print_args
from mmcl.encoder import MMCL_Encoder
from mmcl.losses import MMCL_PGD as MMCL_pgd

parser = argparse.ArgumentParser(description="unsupervised verification")
parser.add_argument(
    "--model",
    type=str,
    default="cnn_4layer_b",
    help="model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)",
)
parser.add_argument("--dataset", default="cifar-10", type=str, help="cifar-10/mnist")
parser.add_argument("--C", default=1.0, type=float, help="C for SVM")
parser.add_argument("--kernel_type", default="rbf", type=str, help="Kernel Type")
parser.add_argument("--num_iters", default=500, type=int, help="Num iters - PGD Solver")
parser.add_argument("--eta", default=1e-5, type=float, help="Eta - PGD Solver")
parser.add_argument(
    "--lr", default=1e-3, type=float, help="learning rate for MMCL encoder"
)
parser.add_argument(
    "--criterion_to_use",
    default="mmcl_pgd",
    type=str,
    help="choose which mmcl svm solver to use",
)
parser.add_argument(
    "--use_norm", default="false", type=str, help="Use Norm - PGD Solver"
)
parser.add_argument("--kernel_gamma", type=str, default="auto")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument(
    "--encoder_checkpoint", default="", type=str, help="PATH TO CHECKPOINT"
)
parser.add_argument(
    "--linear_eval_checkpoint", default="", type=str, help="PATH TO CHECKPOINT"
)
parser.add_argument(
    "--solver_type", default="nesterov", type=str, help="Type of PGD Solver"
)
parser.add_argument(
    "--color_jitter_strength",
    default=0.5,
    type=float,
    help="0.5 for CIFAR, 1.0 for ImageNet",
)
parser.add_argument(
    "--train_type",
    default="contrastive",
    type=str,
    help="contrastive/linear eval/test/supervised",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
print_args(args)

crit = MMCL_pgd(
    sigma=args.kernel_gamma,
    batch_size=args.batch_size,
    anchor_count=2,
    C=args.C,
    solver_type=args.solver_type,
    use_norm=args.use_norm,
    device=device,
    kernel=args.kernel_type,
    eta=args.lr,
)

_, _, _, _, testloader, testdst = (
    data_loader.get_train_val_test_dataset(args)
)

def generate_adversarial_sample(x, loss_function, lr=0.1, num_iterations=50):
    # Ensure x requires gradient
    x = x.clone().detach().requires_grad_(True)

    # Define optimizer to update only x
    optimizer = torch.optim.SGD([x], lr=lr)

    for _ in range(num_iterations):
        optimizer.zero_grad()  # Clear previous gradients
        loss = loss_function(x)  # Compute loss
        (-loss).backward()  # Compute gradient
        optimizer.step()  # Update x

    return x.detach()  # Return the optimized adversarial sample without gradient info

encoder = torch.load(args.encoder_checkpoint, device)
linear_eval = torch.load(args.linear_eval_checkpoint, device)
total_correct, total_samples = 0, 0
for images, labels in testloader:
    print(images.shape)
    # get sample and adv sample
    x = torch.cat(encoder(images), dim=1)
    adv_x = generate_adversarial_sample(
        x=x,
        loss_function=crit,
        lr=args.lr,
        num_iterations=50
    )
    # get preds and adv pred
    preds = torch.argmax(linear_eval(x))
    adv_preds = torch.argmax(linear_eval(adv_x))
    # get how many are correct
    correct = (predictions == targets).sum().item()
    total_correct += correct
    total_samples += labels.size(0)

print(f"{total_correct}/{total_samples}")