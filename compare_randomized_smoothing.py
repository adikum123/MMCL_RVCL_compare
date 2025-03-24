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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import rocl.data_loader as data_loader
from randomized_smoothing.core import Smooth

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
parser.add_argument('--train_type', default='contrastive', type=str, help='contrastive/linear eval/test/supervised')
parser.add_argument('--dataset', default='cifar-10', type=str, help='cifar-10/mnist')
parser.add_argument('--name', default='', type=str, help='name of run')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--color_jitter_strength', default=0.5, type=float, help='0.5 for CIFAR, 1.0 for ImageNet')
parser.add_argument("--picks_per_class", type=int, default=5, help='number of negative items chosen per class')
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--sigma", type=float, default=0.1, help="Sigma for randomized smoothing")
parser.add_argument("--positives_per_class", type=int, default=5, help='number of negative items chosen per class')
parser.add_argument("--batch", type=int, default=1000, help="batch size")

args = parser.parse_args()

# add random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
_, _, _, _, testloader, testdst = data_loader.get_train_val_test_dataset(args=args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CombinedModel(nn.Module):
    def __init__(self, encoder, eval_):
        super(CombinedModel, self).__init__()
        self.encoder = encoder
        self.eval_ = eval_

    def forward(self, x):
        features = self.encoder(x)
        output = self.eval_(features)
        return output

def load_combined_model(args, model_type):
    if model_type in {"mmcl", "regular_cl"}:
        encoder_ckpt = (
            args.mmcl_checkpoint if model_type == "mmcl"
            else args.regular_cl_checkpoint
        )
        eval_ckpt = f"models/linear_evaluate/linear_{encoder_ckpt.split('/')[-1]}"
        print(f"Encoder: {encoder_ckpt}, eval_ckpt: {eval_ckpt}")
        return CombinedModel(
            encoder=torch.load(encoder_ckpt, device),
            eval_=torch.load(eval_ckpt, device)
        )
    print(f"Loading rvcl")
    return torch.load(args.rvcl_checkpoint, device)

def get_ori_model_predicition(model, x):
    return torch.argmax(model(x.unsqueeze(0)), dim=-1).item()


# load models then create verifiers
mmcl_model = load_combined_model(args, "mmcl")
regular_cl_model = load_combined_model(args, "regular_cl")
rvcl_model = load_combined_model(args, "rvcl")
# create verifiers
mmcl_verifier = Smooth(base_classifier=mmcl_model, num_classes=10, sigma=args.sigma)
regular_cl_verifier = Smooth(base_classifier=regular_cl_model, num_classes=10, sigma=args.sigma)
rvcl_verifier = Smooth(base_classifier=rvcl_model, num_classes=10, sigma=args.sigma)
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
    per_class_sampler[class_name].append((image, label))

# construct postives
picks = defaultdict(list)
for class_name, values in per_class_sampler.items():
    picks[class_name] = random.sample(per_class_sampler[class_name], args.positives_per_class)

rs_raidus = defaultdict(list)
for class_name, values in picks.items():
    print(f"Processing class name: {class_name}")
    for image, label in values:
        image = image.to(device)
        mmcl_prediction, mmcl_radius = mmcl_verifier.certify(
            image, args.N0, args.N, args.alpha, args.batch
        )
        print(f"True label: {label}\tOriginal prediction: {get_ori_model_predicition(mmcl_model, image)}\tMMCL smoothed prediction: {mmcl_prediction}\tMMCL radius: {mmcl_radius}")
        regular_cl_prediction, regular_cl_radius = regular_cl_verifier.certify(
            image, args.N0, args.N, args.alpha, args.batch
        )
        print(f"True label: {label}\tOriginal prediction: {get_ori_model_predicition(regular_cl_model, image)}\tRegular CL smoothed prediction: {mmcl_prediction}\tRegular CL radius: {mmcl_radius}")
        rvcl_prediction, rvcl_radius = rvcl_verifier.certify(
            image, args.N0, args.N, args.alpha, args.batch
        )
        print(f"True label: {label}\tOriginal prediction: {get_ori_model_predicition(rvcl_model, image)}\tRVCL smoothed prediction: {rvcl_prediction}\tRVCL radius: {rvcl_radius}")