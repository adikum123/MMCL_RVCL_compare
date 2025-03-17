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
device = torch.device('cpu')


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
for idx, sample in enumerate(testdst):
    image, _, _, label = sample
    class_name = class_names[label]
    per_class_sampler[class_name].append(image)

# construct postives
positives = defaultdict(list)
for class_name, values in per_class_sampler.items():
    positives[class_name] = random.sample(per_class_sampler[class_name], args.positives_per_class)

def compute_radius(verifier, ori_image, target_image):
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
                    compute_radius(
                        verifier=mmcl_verifier, ori_image=ori_image, target_image=target_image
                    )
                )
                rvcl_robust_radius.append(
                    compute_radius(
                        verifier=rvcl_verifier, ori_image=ori_image, target_image=target_image
                    )
                )
                regular_cl_radius.append(
                    compute_radius(
                        verifier=regular_cl_verifier, ori_image=ori_image, target_image=target_image
                    )
                )
            average_robust_radius[f"{index}|{retry}"].append({
                'mmcl': sum(mmcl_robust_radius) / len(mmcl_robust_radius),
                'rvcl': sum(rvcl_robust_radius) / len(rvcl_robust_radius),
                'regular_cl': sum(regular_cl_radius) / len(regular_cl_radius)
            })
print(f"Average robust radius dict:\n{json.dumps(average_robust_radius, indent=4)}")
per_image_values = defaultdict(list)
for index_retry, values in average_robust_radius.items():
    image_index, retry = index_retry.split("|")
    per_image_values[image_index].extend(values)
print(f"Per image values: {json.dumps(per_image_values, indent=4)}")
per_model_mean_std = defaultdict(list)
for image_index, values in per_image_values.items():
    mmcl_values = [x["mmcl"] for x in values]
    rvcl_values = [x["rvcl"] for x in values]
    regular_cl_values = [x["regular_cl"] for x in values]
    per_model_mean_std[image_index] = {
        "mmcl": (np.mean(mmcl_values), np.std(mmcl_values)),
        "rvcl": (np.mean(rvcl_values), np.std(rvcl_values)),
        "regular_cl": (np.mean(regular_cl_values), np.std(regular_cl_values))
    }
print(f"Per model mean and std:\n{json.dumps(per_model_mean_std)}")

def get_model_name_from_ckpt(ckpt):
    model_name = ckpt.split("/")[-1]
    return model_name[0: model_name.rindex(".")]

# Save dictionary as JSON
with open(f"/plots/robust_radius/{file_name}.json", "w") as f:
    json.dump(per_model_mean_std, f, indent=4)

num_images = len(class_names) * args.positives_per_class
assert len(per_model_mean_std.keys()) == len(class_names) * args.positives_per_class, (
    f"Lengths do not match:\n{len(per_model_mean_std.keys())}\n{len(class_names)},{args.positives_per_class}"
)
mmcl_means, mmcl_stds = [0] * num_images, [0] * num_images
rvcl_means, rvcl_stds = [0] * num_images, [0] * num_images
regular_cl_means, regular_cl_stds = [0] * num_images, [0] * num_images

for image_index, value in average_robust_radius.items():
    if isinstance(image_index, str):
        image_index = int(image_index)
    mmcl_mean, mmcl_std = per_model_mean_std[str(image_index)]["mmcl"]
    rvcl_mean, rvcl_std = per_model_mean_std[str(image_index)]["rvcl"]
    regular_cl_mean, regular_cl_std = per_model_mean_std[str(image_index)]["regular_cl"]

    mmcl_means[image_index] = mmcl_mean
    mmcl_stds[image_index] = mmcl_std
    rvcl_means[image_index] = rvcl_mean
    rvcl_stds[image_index] = rvcl_std
    regular_cl_means[image_index] = regular_cl_mean
    regular_cl_stds[image_index] = regular_cl_std

x = list(range(num_images))

plt.figure(figsize=(12, 6))

# MMCL model
plt.errorbar(x, mmcl_means, yerr=mmcl_stds, fmt='o', color='blue', label="MMCL", capsize=3)

# RVCL model
plt.errorbar(x, rvcl_means, yerr=rvcl_stds, fmt='s', color='green', label="RVCL", capsize=3)

# Regular CL model
plt.errorbar(x, regular_cl_means, yerr=regular_cl_stds, fmt='^', color='red', label="Regular CL", capsize=3)

plt.xlabel("Image Index")
plt.ylabel("Robust Radius (Mean ± Std)")
plt.title("Robust Radius Comparison Across Models")
plt.legend()
plt.grid(True)
plt.show()