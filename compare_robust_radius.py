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
for idx, (class_name, positives) in enumerate(positives.items()):
    print(f'Processing class: {class_name}, already processed: {idx}/{len(class_names)} classes')
    for retry in range(args.num_retries):
        print(f"Processing retry: {retry+1}")
        target_images = [image for k, v in per_class_sampler.items() for image in random.sample(v, args.negatives_per_class)]
        for index, ori_image in enumerate(tqdm(positives)):
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
            average_robust_radius[f"{class_name}|{index}"].append({
                'retry_num': f'retry_{retry+1}',
                'mmcl': sum(mmcl_robust_radius) / len(mmcl_robust_radius),
                'rvcl': sum(rvcl_robust_radius) / len(rvcl_robust_radius),
                'regular_cl': sum(regular_cl_radius) / len(regular_cl_radius)
            })

def extract_mean_std_per_model(average_robust_radius):
    for x in average_robust_radius.values():
        retries[x["retry_num"]].append({
            "mmcl": x["mmcl"],
            "rvcl": x["rvcl"],
            "regular_cl": x["regular_cl"],
        })

    averages_per_retry = defaultdict(list)
    for retry, retry_values in retries.items():
        averages_per_retry[retry].append({
            "mmcl": np.mean(list(x["mmcl"] for x in retry_values)),
            "rvcl": np.mean(list(x["rvcl"] for x in retry_values)),
            "regular_cl": np.mean(list(x["regular_cl"] for x in retry_values))
        })

    per_model_extracted_values = defaultdict(list)
    for retry, values in averages_per_retry.items():
        for item in values:
            for model, average_radius in item.items():
                per_model_extracted_values[model].append(average_radius)

    per_model_mean_std = defaultdict(list)
    for model, values in per_model_extracted_values.items():
        per_model_mean_std[model].append(
            (np.mean(values), np.std(values))
        )

    return per_model_mean_std

def get_model_name_from_ckpt(ckpt):
    model_name = ckpt.split("/")[-1]
    return model_name[0: model_name.rindex(".")]

per_model_mean_std = extract_mean_std_per_model(average_robust_radius)
# Save dictionary as JSON
with open(f"/plots/robust_radius/{file_name}.json", "w") as f:
    json.dump(per_model_mean_std, f, indent=4)

images_and_labels = list(average_robust_radius.keys())
mmcl_means, mmcl_stds = [], []
rvcl_means, rvcl_stds = [], []
regular_cl_means, regular_cl_stds = [], []

for key, value in average_robust_radius.items():
    class_name, image_index = key.split("|")
    per_model_mean_std = extract_mean_std_per_model(value)
    mmcl_mean, mmcl_std = per_model_mean_std["mmcl"]
    rvcl_mean, rvcl_std = per_model_mean_std["rvcl"]
    regular_cl_mean, regular_cl_std = per_model_mean_std["regular_cl"]

    mmcl_means.append(mmcl_mean)
    mmcl_stds.append(mmcl_std)
    rvcl_means.append(rvcl_mean)
    rvcl_stds.append(rvcl_std)
    regular_cl_means.append(regular_cl_mean)
    regular_cl_stds.append(regular_cl_std)

x = np.arange(len(class_labels))  # X positions

plt.figure(figsize=(12, 6))

# MMCL Model (Blue)
plt.scatter(x, mmcl_means, color='blue', marker='o', label="MMCL Mean")
plt.scatter(x, mmcl_stds, color='blue', marker='s', label="MMCL Std")

# RVCL Model (Red)
plt.scatter(x, rvcl_means, color='red', marker='o', label="RVCL Mean")
plt.scatter(x, rvcl_stds, color='red', marker='s', label="RVCL Std")

# Regular CL Model (Green)
plt.scatter(x, regular_cl_means, color='green', marker='o', label="Regular CL Mean")
plt.scatter(x, regular_cl_stds, color='green', marker='s', label="Regular CL Std")

plt.xticks(x, class_labels, rotation=45, ha="right", fontsize=9)
plt.ylabel("Mean & Standard Deviation")
plt.title("Mean & Std Deviation per Model")
plt.legend()
plt.grid(axis='y', linestyle="--", alpha=0.7)
plt.tight_layout()
mmcl_name, rvcl_name, regular_cl_name = (
    get_model_name_from_ckpt(args.mmcl_checkpoint),
    get_model_name_from_ckpt(args.rvcl_checkpoint),
    get_model_name_from_ckpt(args.regular_cl_checkpoint)
)
file_name = f"mmcl_{mmcl_name}_rvcl_{rvcl_name}_regular_cl_{regular_cl_name}"
plt.savefig(f"/plots/robust_radius/{file_name}.png", dpi=300, bbox_inches="tight")
