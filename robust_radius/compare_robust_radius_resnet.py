import argparse
import json
import os
import random
from collections import defaultdict
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

from resnet.resnet_unsupervised_encoder import ResnetEncoder
from robust_radius import RobustRadius

parser = argparse.ArgumentParser(description="unsupervised binary search")

parser.add_argument("--dataset", type=str, default="cifar-10")
parser.add_argument("--norm", type=float, default="inf", help="p norm for epsilon perturbation")
parser.add_argument("--max_steps", type=int, default=200, help="max steps for search")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--negatives_per_class", type=int, default=5, help="number of negative items chosen per class")
parser.add_argument("--positives_per_class", type=int, default=5, help="number of positive items chosen per class")
parser.add_argument("--num_retries", type=int, default=3, help="number of retries to minimize effect of randomness")
args = parser.parse_args()

# add random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device("cpu")

def compute_radius(verifier, ori_image, target_image):
    curr_radius = verifier.verify(ori_image, target_image)
    return curr_radius

models = [
    {
        "encoder_ckpt": "models/resnet/finetune_resnet_barlow_bs_512_lr_0.001.pt",
        "model": "resnet cl"
    },
    {
        "encoder_ckpt": "models/resnet/finetune_adv_resnet_info_nce_bs_512_lr_0.001.pt",
        "model": "resnet adversarial cl"
    },
    {
        "encoder_ckpt": "models/resnet/finetune_resnet_mmcl_rbf_C_1.0_bs_512_lr_0.001.pt",
        "model": "resnet mmcl"
    }
]
for model in models:
    # load encoder
    params_dict = {"resnet_encoder_ckpt": model["encoder_ckpt"], "finetune": True}
    hparams = SimpleNamespace(**params_dict)
    model["encoder"] = ResnetEncoder(hparams=hparams, device=device)
    model["verifier"] = RobustRadius(hparams=args, model_ori=model["encoder"])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
])
dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
class_to_images = defaultdict(list)
for img, label in dataset:
    class_to_images[label].append(img)

# Sample seed (positive) images per class
seed_images = []  # list of (class_label, img)
for cls, imgs in class_to_images.items():
    sampled = random.sample(imgs, args.positives_per_class)
    for img in sampled:
        seed_images.append((cls, img))

# Structure: results[seed_idx] = { model_name: [avg_radius_retry1, avg_radius_retry2, ...] }
results = dict()
total_seeds = len(seed_images)
for seed_idx, (cls, ori_img) in enumerate(seed_images):
    # Initialize per-model list
    model_avgs = {m["model"]: [] for m in models}
    for retry in range(args.num_retries):
        # Build a pool of target images (negatives)
        negatives = []
        for imgs in class_to_images.values():
            negatives.extend(random.sample(imgs, args.negatives_per_class))
        # For each model, compute radii list and then average
        for m in models:
            radii = []
            for tgt_img in negatives:
                radius = m["verifier"].verify(ori_img, tgt_img)
                radii.append(radius)
            avg_radius = sum(radii) / len(radii)
            model_avgs[m["model"]].append(avg_radius)
    results[seed_idx] = model_avgs


# Save dictionary as JSON
save_dir = "radius_results"
# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)
file_name = "-".join([x["model"].replace(" ", "_") for x in models])
with open(os.path.join(save_dir, f"{file_name}.json"), "w") as f:
    json.dump(results, f, indent=4)

print(f"Results dict:\n{json.dumps(results, indent=4)}")