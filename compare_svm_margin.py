# Compute margin for mmcl and rvcl encoders using test samples
import argparse
import json
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import ReLU, Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
from torch.nn.modules.padding import ZeroPad2d
from tqdm import tqdm

import mmcl.utils as utils
import rocl.data_loader as data_loader
from beta_crown.modules import Flatten
from mmcl.utils import Flatten as MMCLFlatten
from svm_margin import compute_margin

# arg parser and device
parser = argparse.ArgumentParser(description="Margin comparisson")

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
    "--regular_cl_model", type=str, default=""
)
parser.add_argument(
    "--regular_cl_checkpoint", type=str, default=""
)
parser.add_argument("--train_type", default="test", type=str, help="Must be test")
parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
parser.add_argument("--dataset", default="cifar-10", type=str, help="cifar-10/mnist")
parser.add_argument("--C", type=float, default=1.0, help="C value for SVM algorithm")
parser.add_argument("--kernel_type", default="rbf", type=str, help="Kernel Type")
parser.add_argument(
    "--deegre", default=3.0, type=float, help="Degree for polynomial kernel"
)
parser.add_argument("--seed", default=1, type=int, help="random seed")
parser.add_argument(
    "--color_jitter_strength",
    default=0.0,
    type=float,
    help="0.5 for CIFAR, 1.0 for ImageNet",
)
parser.add_argument(
    "--class_sample_limit",
    default=200,
    type=int,
    help="NUmber of random samples per class",
)
parser.add_argument("--kernel_gamma", type=str, default="auto")
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--positives_per_class", type=int, default=10)
parser.add_argument("--negatives_per_class", type=int, default=100)

args = parser.parse_args()

if args.kernel_gamma not in {'scale', 'auto'}:
    args.kernel_gamma = float(args.kernel_gamma)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load test dataset
_, _, _, _, testloader, testdst = data_loader.get_train_val_test_dataset(args)
class_names = testdst.classes
print("Class names:", class_names)
per_class_sampler = defaultdict(list)
margins = defaultdict(list)

# get class sampler
for idx, sample in enumerate(testdst):
    image, _, _, label = sample
    class_name = class_names[label]
    per_class_sampler[class_name].append(image)

# construct postives
positives = defaultdict(list)
for class_name, values in per_class_sampler.items():
    positives[class_name] = random.sample(per_class_sampler[class_name], args.positives_per_class)


# load both models
mmcl_model = torch.load(args.mmcl_checkpoint, device)
rvcl_model = torch.load(args.rvcl_checkpoint, device)
regular_cl_model = torch.load(args.regular_cl_checkpoint, device)
print(f"Loaded MMCL model: {mmcl_model}")
print(f"Loaded RVCL model: {rvcl_model}")
print(f"Loaded regular cl model: {regular_cl_model}")

def encode_inputs_and_compute_margin(model, positive, negatives):
    positive_encoding = model(positive.unsqueeze(0).to(device))
    negative_encodings = torch.stack([model(neg.unsqueeze(0)) for neg in negatives]).squeeze(1)
    return compute_margin(positive=positive_encoding, negatives=negative_encodings, args=args)

# compute margin for each class
for class_name in class_names:
    print(f"Processing items for class: {class_name}")
    for item in tqdm(per_class_sampler[class_name]):
        # Select one random image as positive and other images as negatives
        positive = item
        for retry in args.num_retries:
            negatives = [image for k, v in per_class_sampler.items() for image in random.sample(v, args.negatives_per_class)]
            mmcl_margin = encode_inputs_and_compute_margin(model=mmcl_model, positive=positive, negatives=negatives)
            rvcl_margin = encode_inputs_and_compute_margin(model=rvcl_model, positive=positive, negatives=negatives)
            regular_cl_margin = encode_inputs_and_compute_margin(model=regular_cl_model, positive=positive, negatives=negatives)
            margins[class_name].append({
                "retry": retry+1,
                "mmcl": mmcl_margin,
                "rvcl": rvcl_margin,
                "regular_cl": regular_cl_margin
            })

print(f"Obtained margin dict: {margins}")
# get mean and std per class
per_class_model_std = {}
for class_name, values in margins:
    mmcl_values = [x["mmcl"] for x in values]
    rvcl_values = [x["rvcl"] for x in values]
    regular_cl_values = [x["regular_cl"] for x in values]
    per_class_model_std[class_name] = {
        "mmcl": (
            np.mean(mmcl_values),
            np.std(mmcl_values)
        ),
        "rvcl": (
            np.mean(rvcl_values),
            np.std(rvcl_values)
        ),
        "regular_cl": (
            np.mean(regular_cl_values),
            np.std(regular_cl_values)
        )
    }


# save all plots
save_dir = f"plots/svm_margin/mmcl_{args.mmcl_model}_rvcl_{args.rvcl_model}_regular_cl_{args.regular_cl_model}_kernel_type_{args.kernel_type}_C_{args.C}"
if args.kernel_type == 'rbf':
    save_dir += f"_gamma_{args.kernel_gamma}"
elif args.kernel_type == 'poly':
    save_dir += f"_deegre_{args.deegre}"
os.makedirs(save_dir, exist_ok=True)
# Loop through classes
for class_name in tqdm(class_names):
    mmcl_values = [x["mmcl"] for x in margins[class_name]]
    rvcl_values = [x["rvcl"] for x in margins[class_name]]
    regular_cl_values = [x["regular_cl"] for x in margins[class_name]]
    print(f"MMCL {class_name} {list(set(mmcl_values))}")
    print(f"RVCL {class_name} {list(set(rvcl_values))}")
    print(f"Regular CL {class_name} {list(set(regular_cl_values))}")
    # Generate x-axis labels (Image {i})
    image_indices = np.arange(len(mmcl_values))
    image_labels = [f"Image {i}" for i in image_indices]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.scatter(image_indices, mmcl_values, color="blue", label="MMCL", alpha=0.7)
    plt.scatter(image_indices, rvcl_values, color="red", label="RVCL", alpha=0.7)
    plt.scatter(image_indices, regular_cl_values, color="orange", label="Regular CL", alpha=0.7)
    # Customize plot
    plt.xticks(image_indices[::max(len(image_indices)//20, 1)], image_labels[::max(len(image_indices)//20, 1)], rotation=45, ha="right")
    plt.legend(loc="upper right")
    plt.title(f"Margin Comparison for {class_name}")
    plt.xlabel("Image Index")
    plt.ylabel("Margin Value")
    plot_path = os.path.join(save_dir, f"{class_name}_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

save_dict = {
    **vars(args),
    "margins": margins
}
file_name = f"mmcl_{args.mmcl_model}_rvcl_{args.rvcl_model}_regular_cl_{args.regular_cl_model}_kernel_type_{args.kernel_type}_C_{args.C}"
if args.kernel_type == 'rbf':
    file_name += f"_gamma_{args.kernel_gamma}"
elif args.kernel_type == 'poly':
    file_name += f"_deegre_{args.deegre}"
# Ensure the directory exists
save_dir = "margin_results"
os.makedirs(save_dir, exist_ok=True)

# Construct file path
file_path = os.path.join(save_dir, f"{file_name}.json")

# Save dictionary as JSON
with open(file_path, "w") as f:
    json.dump(save_dict, f, indent=4)

print(f"Saved results to {file_path}")