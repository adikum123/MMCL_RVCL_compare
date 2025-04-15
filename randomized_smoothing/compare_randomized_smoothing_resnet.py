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
import torchvision
import torchvision.models as vision_models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from randomized_smoothing.core import Smooth
from resnet.resnet import ResNet50
from resnet.resnet_unsupervised import ResnetUnsupervised, SimCLRModel

parser = argparse.ArgumentParser(description="Randomized smoothing ResNet")
parser.add_argument("--batch_size", type=int, default=256, help='batch size')
parser.add_argument("--seed", default=1, type=int, help="random seed")
parser.add_argument("--color_jitter_strength", default=0.5, type=float, help="0.5 for CIFAR, 1.0 for ImageNet")
parser.add_argument("--picks_per_class", type=int, default=5, help="number of negative items chosen per class")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=1000000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--num_images", type=int, default=1000, help="number of images to use")

args = parser.parse_args()

# add random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = [
    {
        "encoder_ckpt": "models/resnet/resnet_supervised_bs_256_lr_0.001.pt",
        "load_classifier": False,
        "model": "resnet supervised"
    },
    {
        "encoder_ckpt": "models/resnet_pretrained_models/resnet50_imagenet_bs2k_epochs600.pth.tar",
        "load_classifier": True,
        "model": "resnet ssl no finetune"
    },
    {
        "encoder_ckpt": "models/resnet_pretrained_models/finetune_1_resnet50_cifar10_bs1024_epochs1000.pth.tar",
        "load_classifier": True,
        "model": "resnet ssl 1 layer finetune"
    },
    {
        "encoder_ckpt": "models/resnet_pretrained_models/finetune_2_resnet50_cifar10_bs1024_epochs1000.pth.tar",
        "load_classifier": True,
        "model": "resnet ssl 2 layer finetune"
    }
]
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
)
testloader = DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False, num_workers=2
)
all_test_images = []
# randomly sample images from the test set
for images, labels in testloader:
    for i in range(images.size(0)):
        all_test_images.append((images[i], labels[i].item()))
picks = random.sample(all_test_images, args.num_images)

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




for config in models:
    encoder_ckpt = config["encoder_ckpt"]
    load_classifier = config["load_classifier"]
    model_name = config["model"]
    if not load_classifier:
        print(f"Loading supervised model: {model_name}")
        # Create supervised model architecture
        model = vision_models.resnet50(weights=vision_models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(2048, 10)
        model.load_state_dict(torch.load(encoder_ckpt, map_location=device))
        model = model.to(device)
        model.eval()
        encoder = model
        config["base_verifier"] = encoder
        continue
    print(f"Loading SimCLR encoder: {model_name}")
    encoder = SimCLRModel(checkpoint_path=encoder_ckpt).to(device)
    encoder.set_eval()
    classifier_ckpt = os.path.join(
        "models", "linear_evaluate", f"linear_{os.path.basename(encoder_ckpt)}"
    )
    print(f"Loading classifier from: {classifier_ckpt}")
    classifier = torch.nn.Sequential(
        torch.nn.Linear(128, 10)
    ).to(device)
    classifier.load_state_dict(torch.load(classifier_ckpt, map_location=device))
    classifier.eval()
    config["base_verifier"] = CombinedModel(encoder=encoder, eval_=classifier)





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
                verifier=unsupervised_verifier, ori_model=resnet_unsupervised, results=results, model_name="unsupervised", true_label=label, image=image
            )

with open(f"rs_results/rs_resnet.json", "w") as f:
    json.dump(results, f)
