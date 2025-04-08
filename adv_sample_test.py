import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import rocl.data_loader as data_loader
from base_encoder import BaseEncoder
from beta_crown.utils import print_args
from mmcl.encoder import MMCL_Encoder
from mmcl.utils import min_max_value
from rocl.attack_lib import FastGradientSignUntargeted

parser = argparse.ArgumentParser(description="unsupervised verification")
parser.add_argument("--dataset", default="cifar-10", type=str, help="cifar-10/mnist")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument(
    "--color_jitter_strength",
    default=0.5,
    type=float,
    help="0.5 for CIFAR, 1.0 for ImageNet",
)
parser.add_argument(
    "--train_type",
    default="linear_eval",
    type=str,
    help="linear eval",
)
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
parser.add_argument("--attack_type", type=str, default="linf", help="adversarial l_p")
parser.add_argument("--random_start", type=bool, default=True, help="True for PGD")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
print_args(args)

_, _, _, _, testloader, testdst = (
    data_loader.get_train_val_test_dataset(args)
)
print(testloader)
models = [
    # regular_cl_model
    {
        "encoder_ckpt": "models/regular_cl/regular_cl_cnn_4layer_b_bs_32_lr_1e-3.pkl",
        "linear_eval_ckpt": "models/linear_evaluate/linear_regular_cl_cnn_4layer_b_bs_32_lr_1e-3.pkl",
        "total_correct": 0,
        "total_samples": 0
    },
    # two mmcl models
    {
        "encoder_ckpt": "models/mmcl/rbf/cnn_4layer_b_C_0.1_rbf_auto.pkl",
        "linear_eval_ckpt": "models/linear_evaluate/linear_cnn_4layer_b_C_0.1_rbf_auto.pkl",
        "total_correct": 0,
        "total_samples": 0
    },
    {
        "encoder_ckpt": "models/mmcl/rbf/cnn_4layer_b_C_1_rbf_auto.pkl",
        "linear_eval_ckpt": "models/linear_evaluate/linear_cnn_4layer_b_C_1_rbf_auto.pkl",
        "total_correct": 0,
        "total_samples": 0
    },
    {
        "encoder_ckpt": "models/mmcl/rbf/cnn_4layer_b_C_10_rbf_auto.pkl",
        "linear_eval_ckpt": "models/linear_evaluate/linear_cnn_4layer_b_C_10_rbf_auto.pkl",
        "total_correct": 0,
        "total_samples": 0
    },
    {
        "encoder_ckpt": "models/mmcl/rbf/cnn_4layer_b_C_100_rbf_auto.pkl",
        "linear_eval_ckpt": "models/linear_evaluate/linear_cnn_4layer_b_C_100_rbf_auto.pkl",
        "total_correct": 0,
        "total_samples": 0
    },
    {
        "encoder_ckpt": "models/mmcl/rbf/cnn_4layer_b_C_1000_rbf_auto.pkl",
        "linear_eval_ckpt": "models/linear_evaluate/linear_cnn_4layer_b_C_1000_rbf_auto.pkl",
        "total_correct": 0,
        "total_samples": 0
    },
    # rvcl adversarial models
    {
        "encoder_ckpt": "models/unsupervised/cifar10_cnn_4layer_b_adv2.pkl",
        "linear_and_encoder_ckpt": "models/linear_evaluate/cifar10_cnn_4layer_b_adv2.pkl",
        "total_correct": 0,
        "total_samples": 0
    },
    {
        "encoder_ckpt": "models/unsupervised/cifar10_cnn_4layer_b_adv4.pkl",
        "linear_and_encoder_ckpt": "models/linear_evaluate/cifar10_cnn_4layer_b_adv4.pkl",
        "total_correct": 0,
        "total_samples": 0
    },
    {
        "encoder_ckpt": "models/unsupervised/cifar10_cnn_4layer_b_adv8.pkl",
        "linear_and_encoder_ckpt": "models/linear_evaluate/cifar10_cnn_4layer_b_adv8.pkl",
        "total_correct": 0,
        "total_samples": 0
    }
]
img_clip = min_max_value(args)

for model_dict in models:
    model_dict["encoder"] = BaseEncoder(ckpt=model_dict["encoder_ckpt"], device=device)
    if "linear_eval_ckpt" in model_dict:
        model_dict["linear_eval"] = torch.load(
            model_dict["linear_eval_ckpt"],
            device
        )
    if "linear_and_encoder_ckpt" in model_dict:
        model_dict["encoder_and_linear_eval"] = torch.load(
            model_dict["linear_and_encoder_ckpt"],
            device
        )
    model_dict["attacker"] = FastGradientSignUntargeted(
        model_dict["encoder"],
        linear="None",
        epsilon=args.epsilon,
        alpha=args.alpha,
        min_val=img_clip["min"].to(device),
        max_val=img_clip["max"].to(device),
        max_iters=args.k,
        device=device,
        _type=args.attack_type,
    )

def get_model_name_from_ckpt(ckpt):
    model_name = ckpt.split("/")[-1]
    return model_name[0: model_name.rindex(".")]

for images, targets in tqdm(testloader):
    for model_dict in models:
        adv_images = model_dict["attacker"].perturb(
            original_images=images,
            labels=targets,
            random_start=args.random_start,
        )
        # get preds and adv pred
        if "linear_eval_ckpt" in model_dict:
            preds = torch.argmax(
                model_dict["linear_eval"](model_dict["encoder"](images)),
                dim=1
            )
            adv_preds = torch.argmax(
                model_dict["linear_eval"](model_dict["encoder"](adv_images)),
                dim=1
            )
        else:
            preds = torch.argmax(
                model_dict["encoder_and_linear_eval"](images),
                dim=1
            )
            adv_preds = torch.argmax(
                model_dict["encoder_and_linear_eval"](adv_images),
                dim=1
            )
        # get how many are correct
        correct = (preds == adv_preds).sum().item()
        model_dict["total_correct"] += correct
        model_dict["total_samples"] += targets.size(0)


for model_dict in models:
    model_name = get_model_name_from_ckpt(model_dict["encoder_ckpt"])
    percentage_unchanged = 100 * (model_dict["total_correct"] / model_dict["total_samples"])
    print(f"Model: {model_name}, unchanged percentage: {percentage_unchanged:.2f}%")


model_names = []
percentages = []

for model_dict in models:
    model_name = get_model_name_from_ckpt(model_dict["encoder_ckpt"])
    percentage_unchanged = 100 * (model_dict["total_correct"] / model_dict["total_samples"])
    model_names.append(model_name)
    percentages.append(percentage_unchanged)

# Create bar plot
plt.figure(figsize=(10, 5))
plt.bar(model_names, percentages, color='skyblue')
plt.xlabel("Model Name")
plt.ylabel("Unchanged Percentage (%)")
plt.title(f"Adversarial Robustness of Models with l_inf norm for eps: {args.epsilon}")
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)

# Annotate bars with values
for i, v in enumerate(percentages):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=10)

plt.tight_layout()
save_dir = "plots/adv_sample"
os.makedirs(save_dir, exist_ok=True)
file_name = f"adv_samples_eps_{args.epsilon}.png"
plt.savefig(os.path.join(save_dir, file_name), dpi=300, bbox_inches="tight")
plt.show()