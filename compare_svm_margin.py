# Compute margin for mmcl and rvcl encoders using test samples
import argparse
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import torch

import mmcl.utils as utils
import rocl.data_loader as data_loader
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
parser.add_argument("--train_type", default="test", type=str, help="Must be test")
parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
parser.add_argument("--dataset", default="cifar-10", type=str, help="cifar-10/mnist")
parser.add_argument("--C", type=float, default=1.0, help="C value for SVM algorithm")
parser.add_argument("--kernel_type", default="rbf", type=str, help="Kernel Type")
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

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load test dataset
_, _, testloader, testdst = data_loader.get_dataset(args)
class_names = testdst.classes
print("Class names:", class_names)
per_class_sampler = defaultdict(list)
margins = defaultdict(list)

# create per class sampler
for idx, (image_batch, label_batch) in enumerate(testloader):
    stop = False
    for image, label in zip(image_batch, label_batch):
        class_name = class_names[label]
        if len(per_class_sampler[class_name]) < args.class_sample_limit:
            per_class_sampler[class_name].append(image)
            if all(
                len(images) >= args.class_sample_limit
                for images in per_class_sampler.values()
            ):
                stop = True
            if stop:
                break
        if stop:
            break

# load both models
mmcl_model = utils.load_model_contrastive_test(
    model=args.mmcl_model, model_path=args.mmcl_checkpoint, device=device
)
rvcl_model = utils.load_model_contrastive_test(
    model=args.rvcl_model, model_path=args.rvcl_checkpoint, device=device
)
print(f"Loaded MMCL model: {mmcl_model}")
print(f"Loaded RVCL model: {rvcl_model}")

# compute margin for each class
for class_name in class_names:
    for item in per_class_sampler[class_name]:
        # Select one random image as positive and other images as negatives
        positive = random.choice(per_class_sampler[class_name])
        while torch.equal(positive, item):
            positive = random.choice(per_class_sampler[class_name])
        negatives = [
            image
            for k, v in per_class_sampler.items()
            for image in v
            if k != class_name
        ]
        # Add batch dimension to positive
        positive = positive.unsqueeze(0)  # Shape: [1, 3, 32, 32]
        # Compute MMCL margin
        mmcl_positive_encoding = mmcl_model(positive)
        mmcl_negative_encodings = torch.stack(
            [mmcl_model(neg.unsqueeze(0)) for neg in negatives]
        ).squeeze(1)
        mmcl_margin = compute_margin(
            positive=mmcl_positive_encoding,
            negatives=mmcl_negative_encodings,
            args=args,
        )
        # Compute RVCL margin
        rvcl_positive_encoding = rvcl_model(positive)
        rvcl_negative_encodings = torch.stack(
            [rvcl_model(neg.unsqueeze(0)) for neg in negatives]
        ).squeeze(1)
        rvcl_margin = compute_margin(
            positive=rvcl_positive_encoding,
            negatives=rvcl_negative_encodings,
            args=args,
        )
        margins[class_name].append({"mmcl": mmcl_margin, "rvcl": rvcl_margin})

# save all plots
save_dir = f"plots/svm_margin/mmcl:{args.mmcl_model}_rvcl:{args.rvcl_model}"
os.makedirs(save_dir, exist_ok=True)
for class_name in class_names:
    mmcl_values = [x["mmcl"] for x in margins[class_name]]
    rvcl_values = [x["rvcl"] for x in margins[class_name]]
    # plot both distributions
    # Plot distributions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(mmcl_values, label="MMCL", fill=True, alpha=0.6)
    sns.kdeplot(rvcl_values, label="RVCL", fill=True, alpha=0.6)

    # Add title and labels
    plt.title(f"Distributions of MMCL and RVCL for Class: {class_name}", fontsize=14)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(title="Metric", fontsize=10)

    # Save the plot
    plot_path = os.path.join(save_dir, f"{class_name}_distribution.png")
    plt.savefig(plot_path)
    plt.close()  # Close the plot to avoid overlap in the next iteration

    print(f"Saved plot for class {class_name} at {plot_path}")
