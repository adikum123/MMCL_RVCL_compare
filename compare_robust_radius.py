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
##### arguments for model #####
parser.add_argument('--train_type', default='contrastive', type=str, help='contrastive/linear eval/test/supervised')
parser.add_argument('--dataset', default='cifar-10', type=str, help='cifar-10/mnist')
parser.add_argument('--load_checkpoint', default='', type=str, help='PATH TO CHECKPOINT')
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
parser.add_argument("--class_sample_limit", type=int, default=5, help='max number of items to compare')

args = parser.parse_args()


# add random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

print("Obtaining data")
# get data
_, _, _, _, testloader, testdst = data_loader.get_train_val_test_dataset(args=args)
print("Obtained data")

# creating verifiers
mmcl_verifier = RobustRadius(hparams=args, model_type='mmcl')
rvcl_verifier = RobustRadius(hparams=args, model_type='mmcl')
print("Loaded verifiers")
# creating data
class_names = testdst.classes
per_class_sampler = defaultdict(list)

print("Iterating through the test dataset")
stop = False  # Flag to stop early if all classes are sampled

for idx in range(len(testdst)):  # Iterate directly through dataset indices
    image, _, _, label = testdst[idx]  # Unpack sample from dataset

    class_name = class_names[label]  # Get class name from label
    if len(per_class_sampler[class_name]) < args.class_sample_limit:
        per_class_sampler[class_name].append(image)

        # Check if we have reached the limit for all classes
        if all(len(images) >= args.class_sample_limit for images in per_class_sampler.values()):
            stop = True
            break  # Stop iterating once all classes are sampled

print("Constructed class sampler")

result_storage = {}

def compute_radius_and_update_storage(verifier, ori_image, target_image):
    key = (
        verifier,
        tuple(ori_image.cpu().numpy().flatten()),
        tuple(target_image.cpu().numpy().flatten())
    )
    reverse_key = (
        verifier,
        tuple(target_image.cpu().numpy().flatten()),
        tuple(ori_image.cpu().numpy().flatten())
    )
    # Check if result already exists in storage
    if key in result_storage:
        return result_storage[key]
    if reverse_key in result_storage:
        return result_storage[reverse_key]
    # Compute the robust radius
    curr_radius = verifier.verify(ori_image, target_image)
    print(f"Computed robust radius: {curr_radius}")
    # Store the computed result
    result_storage[key] = curr_radius
    result_storage[reverse_key] = curr_radius  # Store for both orderings
    return curr_radius

# for each sample in class we compute the average radius
average_robust_radius = defaultdict(list)
for class_name in tqdm(per_class_sampler):
    for ori_image in per_class_sampler[class_name]:
        target_images = [image for k, v in per_class_sampler.items() for image in v if k != class_name]
        mmcl_robust_radius = []
        rvcl_robust_radius = []
        for target_image in target_images:
            print('Computing MMCL robust radius')
            mmcl_robust_radius.append(
                compute_radius_and_update_storage(
                    verifier=mmcl_verifier, ori_image=ori_image, target_image=target_image
                )
            )
            print('Computing RVCL robust radius')
            rvcl_robust_radius.append(
                compute_radius_and_update_storage(
                    verifier=rvcl_verifier, ori_image=ori_image, target_image=target_image
                )
            )
        average_robust_radius[class_name].append({
            'mmcl': sum(mmcl_robust_radius) // len(mmcl_robust_radius),
            'rvcl': sum(rvcl_robust_radius) // len(rvcl_robust_radius)
        })

# save all plots
save_dir = f"plots/robust_radius/mmcl_{args.mmcl_model}_rvcl_{args.rvcl_model}_kernel_type_{args.kernel_type}_C_{args.C}"
if args.kernel_type == 'rbf':
    save_dir += f"_gamma_{args.kernel_gamma}"
elif args.kernel_type == 'poly':
    save_dir += f"_deegre_{args.deegre}"
os.makedirs(save_dir, exist_ok=True)
# Loop through classes
for class_name in tqdm(class_names):
    mmcl_values = [x["mmcl"] for x in average_robust_radius[class_name]]
    rvcl_values = [x["rvcl"] for x in average_robust_radius[class_name]]
    # compute histogram data
    min_value = min(mmcl_values + rvcl_values)
    max_value = max(mmcl_values + rvcl_values)
    bins = np.linspace(min_value, max_value, 100)
    # Create and save plot
    plt.figure()
    plt.hist([mmcl_values, rvcl_values], bins, label=["MMCL", "RVCL"])
    plt.legend(loc="upper right")
    plt.title(f"Margin Distribution for {class_name}")
    plt.xlabel("Margin")
    plt.ylabel("Frequency")
    plot_path = os.path.join(save_dir, f"{class_name}_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()