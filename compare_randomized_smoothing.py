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
parser.add_argument(
    "--supervised_model", type=str, default="", help="model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)"
)
parser.add_argument(
    "--supervised_checkpoint", type=str, default="", help="Model checkpoint for rvcl"
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
parser.add_argument("--positives_per_class", type=int, default=5, help='number of negative items chosen per class')
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--finetune", action="store_true", help="Finetune the model")
parser.add_argument("--relu_layer", action="store_true", help="Use classifier with additional relu layer")
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
        self.encoder.to(device)
        self.eval_.to(device)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
            output = self.eval_(features)
            return output

def load_combined_model(args, model_type):
    assert model_type in {"mmcl", "regular_cl", "rvcl", "supervised"}
    if model_type in {"mmcl", "regular_cl"}:
        encoder_ckpt = (
            args.mmcl_checkpoint if model_type == "mmcl"
            else args.regular_cl_checkpoint
        )
        prefix = ""
        if args.relu_layer:
            prefix += "relu_"
        if args.finetune:
            prefix+= "linear_finetune_"
        eval_ckpt = f"models/linear_evaluate/{prefix}{encoder_ckpt.split('/')[-1].replace('finetune_', '')}"
        print(f"Encoder: {encoder_ckpt}, eval_ckpt: {eval_ckpt}")
        return CombinedModel(
            encoder=torch.load(encoder_ckpt, device, weights_only=False),
            eval_=torch.load(eval_ckpt, device, weights_only=False)
        )
    ckpt = args.rvcl_checkpoint if model_type == "rvcl" else args.supervised_checkpoint
    print(f"Loading model with checkpoint: {ckpt}")
    return torch.load(ckpt, device)

def get_ori_model_predicition(model, x):
    return torch.argmax(model(x.unsqueeze(0)), dim=-1).item()

def update_results(
    verifier, ori_model, results, model_name, true_label, image
):
    rs_label, radius = verifier.certify(
        image, args.N0, args.N, args.alpha, args.batch
    )
    print(f"Radius: {radius}")
    predicted_label = get_ori_model_predicition(ori_model, image)
    results[model_name].append({
        "sigma": verifier.sigma,
        "radius": radius,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "rs_label": rs_label
    })


# load models then create verifiers
mmcl_model = load_combined_model(args, "mmcl")
regular_cl_model = load_combined_model(args, "regular_cl")
rvcl_model = load_combined_model(args, "rvcl")
rvcl_model.to(device)
supervised_model = load_combined_model(args, "supervised")
supervised_model.to(device)
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
results = defaultdict(list)
sigma_values = [0.25, 0.5, 1]
for sigma in sigma_values:
    # create verifiers
    mmcl_verifier = Smooth(base_classifier=mmcl_model, num_classes=10, sigma=sigma)
    regular_cl_verifier = Smooth(base_classifier=regular_cl_model, num_classes=10, sigma=sigma)
    rvcl_verifier = Smooth(base_classifier=rvcl_model, num_classes=10, sigma=sigma)
    supervised_verifier = Smooth(base_classifier=supervised_model, num_classes=10, sigma=sigma)
    print(f"Processing sigma: {sigma}")
    for class_name, values in tqdm(picks.items()):
        for image, label in tqdm(values):
            image = image.to(device)
            update_results(
                verifier=mmcl_verifier, ori_model=mmcl_model, results=results, model_name="mmcl", true_label=label, image=image
            )
            update_results(
                verifier=regular_cl_verifier, ori_model=regular_cl_model, results=results, model_name="regular_cl", true_label=label, image=image
            )
            update_results(
                verifier=rvcl_verifier, ori_model=rvcl_model, results=results, model_name="rvcl", true_label=label, image=image
            )
            update_results(
                verifier=supervised_verifier, ori_model=supervised_model, results=results, model_name="supervised", true_label=label, image=image
            )
results = dict(results)
file_name = f"mmcl_{args.mmcl_model}_rvcl_{args.rvcl_model}_regular_cl_{args.regular_cl_model}_supervised_{args.supervised_model}.json"
with open(f"rs_results/{file_name}.json", "w") as f:
    json.dump(results, f)

certified_radius_choices = [0, 0.5, 1, 1.5, 2, 2.5, 3]
for model_name in ["mmcl", "rvcl", "regular_cl", "supervised"]:
    values = results[model_name]
    for radius in certified_radius_choices:
        for sigma in sigma_values:
            curr_values = [x for x in values if x["sigma"] == sigma and x["radius"] >= radius]
            certified_instance_accuracy, unchanged_percentage = 0, 0
            if len(curr_values) > 0:
                certified_instance_accuracy = sum(1 for x in curr_values if x["true_label"] == x["rs_label"]) / len(curr_values)
                unchanged_percentage = sum(1 for x in curr_values if x["rs_label"] == x["predicted_label"]) / len(curr_values)
                print(f"Model: {model_name}, sigma: {sigma}, radius: {radius}, certified_instance_accuracy: {certified_instance_accuracy}, unchanged_percentage: {unchanged_percentage}")
