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
from torch.utils.data import Subset
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
parser.add_argument("--num_negatives", type=int, default=100)
parser.add_argument("--num_retries", default=5, type=int, help="Number of retries for negative sampling")

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
    image,  label = sample
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
for class_index, class_name in enumerate(class_names):
    print(f"Processing items for class: {class_name} already processed: {class_index}/{len(class_names)} classes")
    for image_index_in_class, item in enumerate(positives[class_name]):
        image_index = class_index * args.positives_per_class + image_index_in_class
        # Select one random image as positive and other images as negatives
        positive = item
        for retry in range(args.num_retries):
            print(f"Processing image: {image_index}, retry: {retry+1}")
            indices = random.sample(range(len(testdst)), args.num_negatives)
            negatives = [testdst[i][0] for i in indices]
            mmcl_margin = encode_inputs_and_compute_margin(model=mmcl_model, positive=positive, negatives=negatives)
            rvcl_margin = encode_inputs_and_compute_margin(model=rvcl_model, positive=positive, negatives=negatives)
            regular_cl_margin = encode_inputs_and_compute_margin(model=regular_cl_model, positive=positive, negatives=negatives)
            margins[f"{image_index}|{retry}"].append({
                "mmcl": mmcl_margin,
                "rvcl": rvcl_margin,
                "regular_cl": regular_cl_margin
            })

per_image_values = defaultdict(list)
for image_index_retry, values in margins.items():
    image_index, retry = image_index_retry.split("|")
    per_image_values[image_index].extend(values)
print(f"Obtained following per_image_values dict")
# get mean and std per class
per_model_mean_std = {}
for image_index, values in per_image_values.items():
    mmcl_values = [x["mmcl"] for x in values]
    rvcl_values = [x["rvcl"] for x in values]
    regular_cl_values = [x["regular_cl"] for x in values]
    per_model_mean_std[image_index] = {
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
    print(f"MMCL values: {mmcl_values}\nmean: {np.mean(mmcl_values)}\nstd: {np.std(mmcl_values)}")
    print(f"RVCL values: {rvcl_values}\nmean: {np.mean(rvcl_values)}\nstd: {np.std(rvcl_values)}")
    print(f"Regular cl values: {regular_cl_values}\nmean: {np.mean(regular_cl_values)}\nstd: {np.std(regular_cl_values)}")
print(f"Obtained following mean and std per class dict:\n{per_model_mean_std}")
# Ensure the directory exists
save_dir = "margin_results"
os.makedirs(save_dir, exist_ok=True)
file_name = f"mmcl_{args.mmcl_model}_rvcl_{args.rvcl_model}_regular_cl_{args.regular_cl_model}_kernel_type_{args.kernel_type}_C_{args.C}"
if args.kernel_type == 'rbf':
    file_name += f"_gamma_{args.kernel_gamma}"
elif args.kernel_type == 'poly':
    file_name += f"_deegre_{args.deegre}"
# Construct file path
file_path = os.path.join(save_dir, f"{file_name}.json")
# Save dictionary as JSON
with open(file_path, "w") as f:
    json.dump(per_model_mean_std, f, indent=4)

# Extract means and standard deviations
mmcl_means = [per_model_mean_std[c]["mmcl"][0] for c in per_model_mean_std.keys()]
mmcl_stds = [per_model_mean_std[c]["mmcl"][1] for c in per_model_mean_std.keys()]

rvcl_means = [per_model_mean_std[c]["rvcl"][0] for c in per_model_mean_std.keys()]
rvcl_stds = [per_model_mean_std[c]["rvcl"][1] for c in per_model_mean_std.keys()]

regular_means = [per_model_mean_std[c]["regular_cl"][0] for c in per_model_mean_std.keys()]
regular_stds = [per_model_mean_std[c]["regular_cl"][1] for c in per_model_mean_std.keys()]

# Improved visualization
plt.figure(figsize=(20, 6))

# Define x-axis positions
image_labels = [f"Image {idx}" for idx in sorted(per_model_mean_std.keys(), key=int)]
x = np.arange(len(image_labels))

# Plot the error bars with smaller markers and thinner lines
plt.errorbar(x, mmcl_means, yerr=mmcl_stds, fmt='o-', markersize=3, linewidth=1, capsize=3, label="MMCL", alpha=0.8)
plt.errorbar(x, rvcl_means, yerr=rvcl_stds, fmt='s-', markersize=3, linewidth=1, capsize=3, label="RVCL", alpha=0.8)
plt.errorbar(x, regular_means, yerr=regular_stds, fmt='d-', markersize=3, linewidth=1, capsize=3, label="Regular CL", alpha=0.8)

# Formatting the plot
plt.xticks(x[::5], image_labels[::5], rotation=45, fontsize=10)  # Show every 5th label for better readability
plt.xlabel("Image")
plt.ylabel("Margin Mean ± Std")
plt.title("SVM Margin Comparison Across Images", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

# Save and show the plot
plt.savefig(os.path.join(save_dir, f"{file_name}.jpg"), bbox_inches="tight", dpi=300)
plt.show()
