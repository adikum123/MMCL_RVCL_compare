import argparse
import copy
import gc
import json
import os
import random
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import rocl.data_loader as data_loader
from beta_crown.modules import Flatten
from mmcl.utils import Flatten as Flatten1
from robust_radius import RobustRadius

parser = argparse.ArgumentParser(description='unsupervised binary search')

##### arguments for CROWN #####
parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
parser.add_argument("--model", type=str, default="cnn_4layer_b", help='model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)')
parser.add_argument("--batch_size", type=int, default=256, help='batch size')
##### mmcl and rvcl args #####
parser.add_argument(
    "--mmcl_model",
    type=str,
    default="",
    help="model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)",
)
parser.add_argument(
    "--mmcl_checkpoint", type=str, default="", help="Model checkpoint for mmcl"
)
parser.add_argument(
    "--rvcl_model",
    type=str,
    default="",
    help="model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)",
)
parser.add_argument(
    "--rvcl_checkpoint", type=str, default="", help="Model checkpoint for rvcl"
)
parser.add_argument(
    "--regular_cl_model",
    type=str,
    default="",
    help="model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)",
)
parser.add_argument(
    "--regular_cl_checkpoint", type=str, default="", help="Model checkpoint for rvcl"
)
##### arguments for model #####
parser.add_argument('--train_type', default='contrastive', type=str, help='contrastive/linear eval/test/supervised')
parser.add_argument('--dataset', default='cifar-10', type=str, help='cifar-10/mnist')
parser.add_argument('--name', default='', type=str, help='name of run')
parser.add_argument('--seed', default=1, type=int, help='random seed')
##### arguments for data augmentation #####
parser.add_argument('--color_jitter_strength', default=0.5, type=float, help='0.5 for CIFAR, 1.0 for ImageNet')
parser.add_argument('--temperature', default=0.5, type=float, help='temperature for pairwise-similarity')
##### arguments for PGD attack & Adversarial Training #####
parser.add_argument('--attack_type', type=str, default='linf', help='adversarial l_p')
parser.add_argument('--target_eps', type=float, default=16.0/255, help='maximum perturbation of adversaries (8/255 0.0314 for cifar-10)')
parser.add_argument('--alpha', type=float, default=0.001, help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
parser.add_argument('--k', type=int, default=150, help='maximum iteration when generating adversarial examples')
parser.add_argument('--random_start', type=bool, default=True, help='True for PGD')
parser.add_argument('--loss_type', type=str, default='mse', help='loss type for Rep: mse/sim/l1/cos')
##### arguments for binary_search #####
parser.add_argument('--mini_batch', type=int, default=10, help='mini batch for PGD')
parser.add_argument('--max_steps', type=int, default=200, help='max steps for search')
parser.add_argument("--negatives_per_class", type=int, default=5, help='number of negative items chosen per class')
parser.add_argument("--positives_per_class", type=int, default=5, help='number of negative items chosen per class')
parser.add_argument("--num_retries", type=int, default=3, help="number of retries to minimize effect of randomness")
args = parser.parse_args()

# add random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
_, _, _, _, testloader, testdst = data_loader.get_train_val_test_dataset(args=args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load models then create verifiers
mmcl_model = torch.load(args.mmcl_checkpoint, device, weights_only=False)
rvcl_model = torch.load(args.rvcl_checkpoint, device, weights_only=False)
regular_cl_model = torch.load(args.regular_cl_checkpoint, device, weights_only=False)

# creating verifiers
mmcl_verifier = RobustRadius(hparams=args, model_ori=mmcl_model)
rvcl_verifier = RobustRadius(hparams=args, model_ori=rvcl_model)
regular_cl_verifier = RobustRadius(hparams=args, model_ori=regular_cl_model)
print("Loaded verifiers")
# creating data
class_names = testdst.classes
per_class_sampler = defaultdict(list)

print("Iterating through the test dataset")
stop = False  # Flag to stop early if all classes are sampled

# get class sampler
for idx, sample in enumerate(testdst):  # Use enumerate to get index and sample directly
    image, _, _, label = sample  # Unpack sample from dataset
    class_name = class_names[label]  # Get class name from label
    per_class_sampler[class_name].append(image)

# construct postives
positives = defaultdict(list)
for class_name, values in per_class_sampler.items():
    positives[class_name] = random.sample(per_class_sampler[class_name], args.positives_per_class)

def compute_radius_and_update_storage(verifier, ori_image, target_image):
    curr_radius = verifier.verify(ori_image, target_image)
    return curr_radius

# for each sample in class we compute the average radius
average_robust_radius = defaultdict(list)
for idx, (class_name, positives) in tqdm(enumerate(positives.items())):
    print(f'Processing class: {class_name}, already processed: {idx}/{len(class_names)} classes')
    for retry in range(args.num_retries):
        target_images = [image for k, v in per_class_sampler.items() for image in random.sample(v, args.negatives_per_class)]
        for index, ori_image in enumerate(positives):
            mmcl_robust_radius = []
            rvcl_robust_radius = []
            regular_cl_radius = []
            for target_image in target_images:
                mmcl_robust_radius.append(
                    compute_radius_and_update_storage(
                        verifier=mmcl_verifier, ori_image=ori_image, target_image=target_image
                    )
                )
                rvcl_robust_radius.append(
                    compute_radius_and_update_storage(
                        verifier=rvcl_verifier, ori_image=ori_image, target_image=target_image
                    )
                )
                regular_cl_radius.append(
                    compute_radius_and_update_storage(
                        verifier=regular_cl_verifier, ori_image=ori_image, target_image=target_image
                    )
                )
            average_robust_radius[f"class_name|{index}"].append({
                'retry_num': f'retry_{retry+1}',
                'mmcl': sum(mmcl_robust_radius) / len(mmcl_robust_radius),
                'rvcl': sum(rvcl_robust_radius) / len(rvcl_robust_radius),
                'regular_cl': sum(regular_cl_radius) / len(regular_cl_radius)
            })

# Convert defaultdict to a structured list
data = []
for key, values in average_robust_radius.items():
    class_name, index = key.split("|")
    index = int(index)  # Convert index to int
    for entry in values:
        data.append({
            "class_name": class_name,
            "index": index,
            "retry_num": entry["retry_num"],
            "mmcl": entry["mmcl"],
            "rvcl": entry["rvcl"],
            "regular_cl": entry["regular_cl"]
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# Compute mean and standard deviation per image for each model
stats_df = df.groupby(["class_name", "index"]).agg(
    mmcl_mean=("mmcl", "mean"),
    mmcl_std=("mmcl", "std"),
    rvcl_mean=("rvcl", "mean"),
    rvcl_std=("rvcl", "std"),
    regular_cl_mean=("regular_cl", "mean"),
    regular_cl_std=("regular_cl", "std")
).reset_index()

# Create formatted x-axis labels
stats_df["label"] = stats_df.apply(lambda row: f"image {row['index']} ({row['class_name']})", axis=1)

# Plot
plt.figure(figsize=(12, 6))
x = np.arange(len(stats_df))  # X-axis positions

plt.errorbar(x, stats_df["mmcl_mean"], yerr=stats_df["mmcl_std"], fmt='o-', capsize=5, label="MMCL", color='blue')
plt.errorbar(x, stats_df["rvcl_mean"], yerr=stats_df["rvcl_std"], fmt='s-', capsize=5, label="RVCL", color='red')
plt.errorbar(x, stats_df["regular_cl_mean"], yerr=stats_df["regular_cl_std"], fmt='^-', capsize=5, label="Regular CL", color='green')

# Formatting
plt.xticks(x, stats_df["label"], rotation=45, ha="right")
plt.xlabel("Images")
plt.ylabel("Robust Radius")
plt.title("Average Robust Radius with Standard Deviation")
plt.legend()
plt.tight_layout()
plt.show()


# save all plots
save_dir = f"plots/robust_radius/mmcl_{args.mmcl_model}_rvcl_{args.rvcl_model}_regular_cl_{args.regular_cl_model}"
os.makedirs(save_dir, exist_ok=True)

save_dict = {
    **vars(args),
    "average_robust_radius": average_robust_radius
}

def get_model_name_from_ckpt(ckpt):
    model_name = ckpt.split("/")[-1]
    return model_name[0: model_name.rindex(".")]


file_name = f"mmcl_{get_model_name_from_ckpt(args.mmcl_checkpoint)}_rvcl_{get_model_name_from_ckpt(args.rvcl_checkpoint)}_regular_cl_{get_model_name_from_ckpt(args.regular_cl_checkpoint)}"
# Ensure the directory exists
save_dir = "radius_results"
os.makedirs(save_dir, exist_ok=True)

# Construct file path
file_path = os.path.join(save_dir, f"{file_name}.json")

# Save dictionary as JSON
with open(file_path, "w") as f:
    json.dump(save_dict, f, indent=4)

print(f"Saved results to {file_path}")