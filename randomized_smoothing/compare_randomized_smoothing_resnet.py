import argparse
import json
import os
import random
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from randomized_smoothing.core import Smooth
from resnet.resnet import ResNet50
from resnet.resnet_supervised import ResnetSupervised
from resnet.resnet_unsupervised_classifier import ResnetUnsupervisedClassifier
from resnet.resnet_unsupervised_encoder import ResnetEncoder

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
        "encoder_ckpt": "models/resnet/finetune_resnet_info_nce_bs_512_lr_0.001.pt",
        "load_classifier": True,
        "model": "resnet cl"
    },
    {
        "encoder_ckpt": "models/resnet/finetune_adv_resnet_info_nce_bs_512_lr_0.001.pt",
        "load_classifier": True,
        "model": "resnet adversarial cl"
    },
    {
        "encoder_ckpt": "models/resnet/finetune_resnet_mmcl_rbf_C_1.0_bs_512_lr_0.001.pt",
        "load_classifier": True,
        "model": "resnet mmcl"
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

def get_test_set_accuracy(model):
    total_correct, total_samples = 0, 0
    for images, targets in testloader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == targets).sum().item()
        total_correct += correct
        total_samples += targets.size(0)
    return total_correct / total_samples

for model in models:
    if not model["load_classifier"]:
        params_dict = {"resnet_supervised_ckpt": model["encoder_ckpt"], "finetune": True}
        hparams = SimpleNamespace(**params_dict)
        model["base_classifier"] = ResnetSupervised(hparams=hparams, device=device)
        continue
    # load encoder
    params_dict = {"resnet_encoder_ckpt": model["encoder_ckpt"], "finetune": True}
    hparams = SimpleNamespace(**params_dict)
    encoder = ResnetEncoder(hparams=hparams, device=device)
    # load classifier
    params_dict = {
        "resnet_classifier_ckpt": os.path.join("models", "linear_evaluate", f"linear_{os.path.basename(model['encoder_ckpt'])}"),
        "finetune": True
    }
    hparams = SimpleNamespace(**params_dict)
    classifier = ResnetUnsupervisedClassifier(hparams=hparams, device=device, encoder=encoder)
    model["base_classifier"] = classifier
    model["test_accuracy"] = get_test_set_accuracy(model["base_classifier"])
    print(f"Test accuracy for model {model['model']}: {model['test_accuracy']}")

results = defaultdict(list)
sigma_values = [0.12, 0.25, 0.5, 0.6]
for sigma in sigma_values:
# create verifiers
    for model in models:
        model["verifier"] = Smooth(
            base_classifier=model["base_classifier"],
            num_classes=10,
            sigma=sigma
        )
    print(f"\nProcessing sigma: {sigma}")
    for model in models:
        for image, label in tqdm(picks):
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
results["models_info"] = [{"model": x["model"], "test_accuracy": x["test_accuracy"]} for x in models]
output_file_name = "-".join([x["model"].replace(" ", "_") for x in models])
output_dir = os.path.join("rs_results")
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, f"{output_file_name}.json"), "w") as f:
    json.dump(results, f, indent=4)
