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
parser.add_argument("--N", type=int, default=1000000, help="number of samples to use")
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

models = [
    {
        "encoder_ckpt": "models/mmcl/rbf/finetune_mmcl_cnn_4layer_b_C_1.0_bs_512_lr_0.0001.pkl",
        "load_classifier": True,
        "model": "mmcl rbf"
    },
    {
        "encoder_ckpt": "models/linear_evaluate/cifar10_cnn_4layer_b_adv8.pkl",
        "load_classifier": False,
        "model": "adversarial cl"
    },
    {
        "encoder_ckpt": "models/regular_cl/finetune_regular_cl_info_nce_bs_512_lr_0.001.pkl",
        "load_classifier": True,
        "model": "cl info_nce"
    },
    {
        "encoder_ckpt": "models/supervised/supervised_clean_bs_256_lr_0.001.pkl",
        "load_classifier": False,
        "model": "supervised"
    }
]

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
    if model_type in {"mmcl", "regular_cl", "rvcl"}:
        encoder_ckpt = (
            args.mmcl_checkpoint if model_type == "mmcl"
            else args.regular_cl_checkpoint
        )
        prefix = ""
        if args.relu_layer:
            prefix += "relu_"
        if args.finetune:
            prefix+= "linear_finetune_clean_"
        eval_ckpt = f"models/linear_evaluate/{prefix}{encoder_ckpt.split('/')[-1].replace('finetune_', '')}"
        print(f"Encoder: {encoder_ckpt}, eval_ckpt: {eval_ckpt}")
        return CombinedModel(
            encoder=torch.load(encoder_ckpt, device, weights_only=False),
            eval_=torch.load(eval_ckpt, device, weights_only=False)
        )
    return torch.load(args.supervised_checkpoint, device)

def get_ori_model_predicition(model, x):
    return torch.argmax(model(x.unsqueeze(0)), dim=-1).item()

def update_results(
    verifier, ori_model, results, model_name, true_label, image
):
    rs_label, radius = verifier.certify(
        image, args.N0, args.N, args.alpha, args.batch
    )
    predicted_label = get_ori_model_predicition(ori_model, image)
    print(f"Model: {model_name}, true_label: {true_label}, predicted_label: {predicted_label}, rs_label: {rs_label}, radius: {radius}")
    results[model_name].append({
        "sigma": verifier.sigma,
        "radius": radius,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "rs_label": rs_label
    })

for model in models:
    encoder = torch.load(model["encoder_ckpt"], map_location=device, weights_only=False)
    prefix = ""
    if args.relu_layer:
        prefix += "relu_"
    if model["load_classifier"]:
        eval_ckpt = f"models/linear_evaluate/{prefix}linear_finetune_clean_{model['encoder_ckpt'].split('/')[-1].replace('finetune_', '')}"
        print(f"Loaded:\nencoder:{model['encoder_ckpt']}\nclassifier:{eval_ckpt}")
        classifier = torch.load(eval_ckpt, map_location=device, weights_only=False)
        model["base_classifier"] = CombinedModel(encoder=encoder, eval_=classifier)
    else:
        print(f"Loaded:\nencoder:{model['encoder_ckpt']}")
        model["base_classifier"] = encoder

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
sigma_values = [0.12, 0.25, 0.5, 0.67, 1]
for sigma in sigma_values:
# create verifiers
    for model in models:
        model["verifier"] = Smooth(
            base_classifier=model["base_classifier"],
            num_classes=10,
            sigma=sigma
        )
    print(f"Processing sigma: {sigma}")
    for class_name, values in picks.items():
        print(f"Processing class: {class_name}")
        for class_name, values in tqdm(picks.items()):
            for image, label in tqdm(values):
                for model in models:
                    image = image.to(device)
                    update_results(
                        verifier=model["verifier"],
                        ori_model=model["base_classifier"],
                        results=results,
                        model_name=model["model"],
                        true_label=label,
                        image=image
                    )
results = dict(results)
def get_name_from_ckpt(ckpt):
    ckpt = ckpt.split("/")[-1]
    return ckpt[0: ckpt.rindex(".")]

file_name = "_".join([x["model"] for x in models])
with open(f"rs_results/{file_name}.json", "w") as f:
    json.dump(results, f, indent=4)
