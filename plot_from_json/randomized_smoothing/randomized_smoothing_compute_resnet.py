import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font

file_name = "resnet_cl_nce-resnet_cl_barlow-resnet_cl_cosine-resnet_cl_info_nce"
with open(f"rs_results/{file_name}.json", "r") as f:
    data = json.load(f)
models_info = data["models_info"]
model_names = [x["model"] for x in models_info]


def plot_one_certified_accuracy_per_sigma(data):
    # Create a mapping from model name to test accuracy
    model_to_accuracy = {m["model"]: m["test_accuracy"] for m in models_info}

    sigma_set = set()
    for model in model_names:
        for rec in data[model]:
            sigma_set.add(rec["sigma"])
    sigma_values = sorted(list(sigma_set))

    for sigma in sigma_values:
        plt.figure(figsize=(8, 6))
        max_threshold = 0
        model_records = {}
        for model in model_names:
            records = [r for r in data[model] if r["sigma"] == sigma]
            model_records[model] = records
            if records:
                max_val = max(r["radius"] for r in records)
                if max_val > max_threshold:
                    max_threshold = max_val
        if max_threshold == 0:
            max_threshold = 3.5

        x_vals = np.linspace(0, max_threshold, 200)
        for model in model_names:
            records = model_records[model]
            total = len(records)
            y_vals = []
            for r in x_vals:
                count = sum(
                    1 for rec in records if rec["radius"] >= r and rec["true_label"] == rec["rs_label"]
                )
                y_vals.append(count / total if total > 0 else 0)
            # Construct legend label with test accuracy
            test_acc = model_to_accuracy.get(model, "N/A")
            label = f"{model} (acc: {100*test_acc:.2f}%)" if isinstance(test_acc, float) else f"{model} (acc: N/A)"
            plt.plot(x_vals, y_vals, label=label)

        plt.xlabel("Radius Threshold")
        plt.ylabel("Certified Accuracy")
        plt.title(f"Certified Accuracy vs Radius (sigma = {sigma})")
        plt.legend()
        plt.grid(True)
        output_dir = os.path.join("plots", "randomized_smoothing", "resnet_cl")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"ca_sigma_{sigma}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, output_filename))
        plt.close()
        print(f"Plot saved as: {output_filename}")


def plot_one_unchanged_percentage_per_sigma(data):
    # Create a mapping from model name to test accuracy
    model_to_accuracy = {m["model"]: m["test_accuracy"] for m in models_info}

    sigma_set = set()
    for model in model_names:
        for rec in data[model]:
            sigma_set.add(rec["sigma"])
    sigma_values = sorted(list(sigma_set))

    for sigma in sigma_values:
        plt.figure(figsize=(8, 6))
        max_threshold = 0
        model_records = {}
        for model in model_names:
            records = [r for r in data[model] if r["sigma"] == sigma]
            model_records[model] = records
            if records:
                max_val = max(r["radius"] for r in records)
                if max_val > max_threshold:
                    max_threshold = max_val
        if max_threshold == 0:
            max_threshold = 3.5

        x_vals = np.linspace(0, max_threshold, 200)
        for model in model_names:
            records = model_records[model]
            total = len(records)
            y_vals = []
            for r in x_vals:
                count = sum(
                    1 for rec in records if rec["radius"] >= r and rec["predicted_label"] == rec["rs_label"]
                )
                y_vals.append(count / total if total > 0 else 0)
            # Construct legend label with test accuracy
            test_acc = model_to_accuracy.get(model, "N/A")
            label = f"{model} (acc: {100*test_acc:.2f}%)" if isinstance(test_acc, float) else f"{model} (acc: N/A)"
            plt.plot(x_vals, y_vals, label=label)

        plt.xlabel("Radius Threshold")
        plt.ylabel("Certified Accuracy")
        plt.title(f"Unchanged Percentage vs Radius (sigma = {sigma})")
        plt.legend()
        plt.grid(True)
        output_dir = os.path.join("plots", "randomized_smoothing", "resnet_cl")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"up_sigma_{sigma}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, output_filename))
        plt.close()
        print(f"Plot saved as: {output_filename}")


plot_one_certified_accuracy_per_sigma(data)
plot_one_unchanged_percentage_per_sigma(data)