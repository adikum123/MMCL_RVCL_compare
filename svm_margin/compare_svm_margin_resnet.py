# Compute margin for mmcl and rvcl encoders using test samples
import argparse
import json
import os
import random
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

import mmcl.utils as utils
import rocl.data_loader as data_loader
from resnet.resnet_unsupervised_encoder import ResnetEncoder
from svm_margin import compute_margin

# arg parser and device
parser = argparse.ArgumentParser(description="Margin comparisson")

parser.add_argument("--train_type", default="test", type=str, help="Must be test")
parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
parser.add_argument("--dataset", default="cifar-10", type=str, help="cifar-10/mnist")
parser.add_argument(
    "--color_jitter_strength",
    default=0.0,
    type=float,
    help="0.5 for CIFAR, 1.0 for ImageNet",
)
parser.add_argument("--C", type=float, default=1.0, help="C value for SVM algorithm")
parser.add_argument("--kernel_type", default="rbf", type=str, help="Kernel Type")
parser.add_argument("--seed", default=1, type=int, help="random seed")
parser.add_argument("--kernel_gamma", type=str, default="auto")
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--positives_per_class", type=int, default=10)
parser.add_argument("--negatives_per_class", type=int, default=100)
parser.add_argument("--num_retries", default=5, type=int, help="Number of retries for negative sampling")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model["encoder"].set_eval()

# load test dataset
_, _, _, _, testloader, testdst = data_loader.get_train_val_test_dataset(args)
class_names = testdst.classes
print("Class names:", class_names)
per_class_sampler = defaultdict(list)
margins = defaultdict(list)

# get class sampler
for idx, sample in enumerate(testdst):
    image,  label = sample
    class_name = class_names[label]
    per_class_sampler[class_name].append(image)

# construct postives
positives = defaultdict(list)
for class_name, values in per_class_sampler.items():
    positives[class_name] = random.sample(per_class_sampler[class_name], args.positives_per_class)


def encode_inputs_and_compute_margin(model, positive, negatives):
    model = model.to(device)
    positive = positive.to(device)
    negatives = [neg.to(device) for neg in negatives]
    positive_encoding = model(positive.unsqueeze(0).to(device))
    negative_encodings = torch.stack([model(neg.unsqueeze(0)) for neg in negatives]).squeeze(1)
    return compute_margin(positive=positive_encoding, negatives=negative_encodings, args=args)

# compute margin for each class
for class_index, class_name in enumerate(class_names):
    print(f"Processing items for class: {class_name} already processed: {class_index}/{len(class_names)} classes")
    for image_index_in_class, item in enumerate(tqdm(positives[class_name])):
        image_index = class_index * args.positives_per_class + image_index_in_class
        # Select one random image as positive and other images as negatives
        positive = item
        for retry in range(args.num_retries):
            result_dct = {}
            for model in models:
                print(f"Processing image: {image_index}, retry: {retry+1}, model: {model['model']}")
                negatives = [
                    image
                    for k, v in per_class_sampler.items()
                    for image in random.sample(v, args.negatives_per_class)
                    if k != class_name
                ]
                margin = encode_inputs_and_compute_margin(model=model["encoder"], positive=positive, negatives=negatives)
                result_dct[model["model"]] = margin
            # Store the results in the margins dictionary
            print(json.dumps(result_dct, indent=4))
            margins[f"{image_index}|{retry}"].append(result_dct)

per_image_values = defaultdict(list)
for image_index_retry, values in margins.items():
    image_index, retry = image_index_retry.split("|")
    per_image_values[image_index].extend(values)
print(f"Obtained following per_image_values dict")
# get mean and std per class
per_model_mean_std = {}
for image_index, values in per_image_values.items():
    for model in models:
        model_values = [v[model["model"]] for v in values]
        mean = np.mean(model_values)
        std = np.std(model_values)
        if image_index not in per_model_mean_std:
            per_model_mean_std[image_index] = {}
        per_model_mean_std[image_index][model["model"]] = (mean, std)
per_model_mean_std ["metadata"] = {
    "positives_per_class": args.positives_per_class,
    "negatives_per_class": args.negatives_per_class,
    "num_retries": args.num_retries,
    "kernel_type": args.kernel_type,
    "kernel_gamma": args.kernel_gamma if "kernel_gamma" in vars(args) else None,
    "C": args.C
}
print(f"Obtained following mean and std per class dict:\n{per_model_mean_std}")
# Ensure the directory exists
save_dir = "margin_results"
os.makedirs(save_dir, exist_ok=True)
file_name = "-".join([x["model"].replace(" ", "_") for x in models])
file_name = f"{file_name}_{args.kernel_type}"
# Construct file path
file_path = os.path.join(save_dir, f"{file_name}.json")
# Save dictionary as JSON
with open(file_path, "w") as f:
    json.dump(per_model_mean_std, f, indent=4)
