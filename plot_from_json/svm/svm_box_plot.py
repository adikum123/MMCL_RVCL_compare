import json
import os

import matplotlib.pyplot as plt

file_name = "resnet_cl-resnet_adversarial_cl-resnet_mmcl"
with open(f"margin_results/{file_name}.json", "r") as f:
    data = json.load(f)

mmcl_values, regular_cl_values, rvcl_values = [], [], []
for image_index, models_values in data.items():
    for model_name, values in models_values.items():
        if model_name == "mmcl":
            mmcl_values.append(values[0])
            continue
        if model_name == "regular_cl":
            regular_cl_values.append(values[0])
            continue
        if model_name == "rvcl":
            rvcl_values.append(values[0])
            continue
        raise ValueError(f"Unknown model name: {model_name}")

# Prepare data for plotting
data_to_plot = [mmcl_values, rvcl_values, regular_cl_values]
labels = ["MMCL", "RVCL", "Regular CL"]
colors = ["blue", "green", "red"]

# Create and configure plot
plt.figure(figsize=(10, 7))
bp = plt.boxplot(data_to_plot,
                patch_artist=True,
                vert=True,
                labels=labels,
                medianprops={"color": "black", "linewidth": 1.5})

# Customize colors
for patch, color in zip(bp["boxes"], colors):
    patch.set(facecolor=color, alpha=0.6)

# Add title and labels
plt.title("Model Performance Comparison", fontsize=14)
plt.ylabel("Values", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

# Save and show plot
plt.savefig(os.path.join("../plots/svm_margin", f"{file_name}_boxplot_comparison.png"), dpi=300, bbox_inches="tight")
plt.show()
