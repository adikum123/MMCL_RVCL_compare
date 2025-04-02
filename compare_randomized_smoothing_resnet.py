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
import torchvision.models as models
from tqdm import tqdm

import rocl.data_loader as data_loader
from randomized_smoothing.core import Smooth

parser = argparse.ArgumentParser(description="unsupervised binary search")
parser.add_argument("--batch_size", type=int, default=256, help='batch size')
parser.add_argument("--train_type", default="contrastive", type=str, help="contrastive/linear eval/test/supervised")
parser.add_argument("--dataset", default="cifar-10", type=str, help="cifar-10/mnist")
parser.add_argument("--name", default="", type=str, help="name of run")
parser.add_argument("--seed", default=1, type=int, help="random seed")
parser.add_argument("--color_jitter_strength", default=0.5, type=float, help="0.5 for CIFAR, 1.0 for ImageNet")
parser.add_argument("--picks_per_class", type=int, default=5, help="number of negative items chosen per class")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--positives_per_class", type=int, default=5, help="number of negative items chosen per class")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--resnet_supervised_ckpt", type=str, help="Checkpoint for supervised resnet model")
parser.add_argument("--resnet_unsupervised_encoder_ckpt", type=str, help="Checkpoint for supervised resnet model")
parser.add_argument("--resnet_unsupervised_eval_ckpt", type=str, help="Checkpoint for supervised resnet model")
args = parser.parse_args()

# add random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
_, _, _, _, testloader, testdst = data_loader.get_train_val_test_dataset(args=args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CombinedResnetModel(nn.Module):
    def __init__(self, encoder_ckpt, eval_ckpt):
        super(CombinedResnetModel, self).__init__()
        # load encoder
        encoder_checkpoint = torch.load(encoder_ckpt, map_location=device)
        state_dict = encoder_checkpoint["state_dict"]
        new_state_dict = {k.replace("convnet.", ""): v for k, v in state_dict.items() if not k.startswith("projection.")}
        self.encoder = models.resnet50(pretrained=False)
        self.encoder.load_state_dict(new_state_dict, strict=True)
        self.encoder.fc = torch.nn.Identity()
        self.encoder.to(device)
        # load classifier
        eval_checkpoint = torch.load(eval_ckpt, map_location=device)
        self.classifier = torch.nn.Linear(2048, 10)
        self.classifier.load_state_dict(eval_checkpoint)
        self.classifier.to(device)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
            output = self.classifier(features)
            return output

def get_ori_model_predicition(model, x):
    return torch.argmax(model(x.unsqueeze(0)), dim=-1).item()

def update_results(
    verifier, ori_model, results, model_name, true_label, image
):
    rs_label, radius = verifier.certify(
        image, args.N0, args.N, args.alpha, args.batch
    )
    predicted_label = get_ori_model_predicition(ori_model, image)
    results[model_name].append({
        "sigma": verifier.sigma,
        "radius": radius,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "rs_label": rs_label
    })

# creating data
class_names = testdst.classes
per_class_sampler = defaultdict(list)


# get class sampler
for idx, sample in enumerate(testdst):
    image, _, _, label = sample
    class_name = class_names[label]
    per_class_sampler[class_name].append((image, label))

# load resnet supervised model
resnet_supervised = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet_supervised.fc = torch.nn.Linear(2048, 10)
checkpoint = torch.load(args.resnet_supervised_ckpt, map_location=device)
resnet_supervised.load_state_dict(checkpoint)
resnet_supervised.to(device)
# load resnet unsupervised model
resnet_unsupervised = CombinedResnetModel(
    encoder_ckpt=args.resnet_unsupervised_encoder_ckpt,
    eval_ckpt=args.resnet_unsupervised_eval_ckpt,
)

# create verifiers
# construct postives
picks = defaultdict(list)
for class_name, values in per_class_sampler.items():
    picks[class_name] = random.sample(per_class_sampler[class_name], args.positives_per_class)
results = defaultdict(list)
sigma_values = [0.12, 0.25, 0.5, 0.67, 1]
for sigma in sigma_values:
    # create verifiers
    supervised_verifier = Smooth(base_classifier=resnet_supervised, num_classes=10, sigma=sigma)
    unsupervised_verifier = Smooth(base_classifier=resnet_unsupervised, num_classes=10, sigma=sigma)
    print(f"Processing sigma: {sigma}")
    for class_name, values in tqdm(picks.items()):
        for image, label in tqdm(values):
            image = image.to(device)
            update_results(
                verifier=supervised_verifier, ori_model=resnet_supervised, results=results, model_name="supervised", true_label=label, image=image
            )
            update_results(
                verifier=unsupervised_verifier, ori_model=resnet_unsupervised, results=results, model_name="regular_cl", true_label=label, image=image
            )

with open(f"rs_results/rs_resnet.json", "w") as f:
    json.dump(results, f)
