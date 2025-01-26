import argparse
import copy
import gc
import json
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch.nn.functional as F

import rocl.data_loader as data_loader
from beta_crown.auto_LiRPA import BoundedModule, BoundedTensor
from beta_crown.auto_LiRPA.perturbations import *
from beta_crown.model_beta_CROWN import LiRPAConvNet, return_modify_model
from beta_crown.relu_conv_parallel import relu_bab_parallel
from beta_crown.utils import *
from robust_radius import RobustRadius
from rocl.attack_lib import FastGradientSignUntargeted, RepresentationAdv

parser = argparse.ArgumentParser(description="unsupervised binary search")

##### arguments for CROWN #####
parser.add_argument(
    "--norm", type=float, default="inf", help="p norm for epsilon perturbation"
)
parser.add_argument(
    "--mmcl_model",
    type=str,
    default="cnn_4layer_b",
    help="model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)",
)
parser.add_argument(
    "--rvcl_model",
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
    "--mmcl_load_checkpoint", default="", type=str, help="PATH TO CHECKPOINT"
)
parser.add_argument(
    "--rvcl_load_checkpoint", default="", type=str, help="PATH TO CHECKPOINT"
)
parser.add_argument("--name", default="", type=str, help="name of run")
parser.add_argument("--seed", default=1, type=int, help="random seed")

##### arguments for data augmentation #####
parser.add_argument(
    "--color_jitter_strength",
    default=0.5,
    type=float,
    help="0.5 for CIFAR, 1.0 for ImageNet",
)
parser.add_argument(
    "--temperature", default=0.5, type=float, help="temperature for pairwise-similarity"
)

##### arguments for PGD attack & Adversarial Training #####
parser.add_argument("--attack_type", type=str, default="linf", help="adversarial l_p")
parser.add_argument(
    "--target_eps",
    type=float,
    default=16.0 / 255,
    help="maximum perturbation of adversaries (8/255 0.0314 for cifar-10)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.001,
    help="movement multiplier per iteration when generating adversarial examples (2/255=0.00784)",
)
parser.add_argument(
    "--k",
    type=int,
    default=150,
    help="maximum iteration when generating adversarial examples",
)
parser.add_argument("--random_start", type=bool, default=True, help="True for PGD")
parser.add_argument(
    "--loss_type", type=str, default="mse", help="loss type for Rep: mse/sim/l1/cos"
)

##### arguments for binary_search #####
parser.add_argument("--mini_batch", type=int, default=10, help="mini batch for PGD")
parser.add_argument(
    "--ver_total", type=int, default=100, help="number of img to verify"
)
parser.add_argument("--max_steps", type=int, default=200, help="max steps for search")
parser.add_argument(
    "--class_sample_limit",
    default=10,
    type=int,
    help="NUmber of random samples per class",
)

args = parser.parse_args()
print_args(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

img_clip = min_max_value(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

verifier = RobustRadius(hparams=args, model_type='mmcl')

def generate_ver_data(loader, total, class_num, adv=True):
    count = [0 for _ in range(class_num)]
    per_class = total // class_num
    data_loader = iter(loader)
    ans_image = []
    if adv:
        adv_target = []
        adv_eps = []
    ans_label = []
    while sum(count) < total:
        (ori, aug_img, _, label) = next(data_loader)
        i = int(label)
        if count[i] < per_class:
            ans_image.append(ori)
            ans_label.append(i)
            if adv:
                i1, i2 = generate_attack(args, ori, aug_img)
                adv_target.append(i1)
                adv_eps.append(i2)
            count[i] += 1
    if adv:
        return ans_image, adv_target, adv_eps, ans_label
    else:
        return ans_image, ans_label


# Data
print("==> Preparing data..")
_, _, testloader, testdst = data_loader.get_dataset(args)
class_names = testdst.classes
image, label = generate_ver_data(testloader, args.ver_total, class_num=10, adv=False)
per_class_sampler = defaultdict(list)

# create per class sampler
for idx, (image_batch, _, _, label_batch) in enumerate(testloader):
    stop = False
    for image, label in zip(image_batch, label_batch):
        class_name = class_names[label]
        if len(per_class_sampler[class_name]) < args.class_sample_limit:
            per_class_sampler[class_name].append(image)
            if all(
                len(images) >= args.class_sample_limit
                for images in per_class_sampler.values()
            ):
                stop = True
            if stop:
                break
        if stop:
            break

sample1 = per_class_sampler[class_names[random.randint(0, 9)]][0]
sample2 = per_class_sampler[class_names[random.randint(0, 9)]][0]
verifier.verify(sample1, sample2)