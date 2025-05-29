import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

file_name = "resnet_cl-resnet_adversarial_cl-resnet_mmcl_rbf"
input_path = f"margin_results/{file_name}.json"
output_dir = "plots/svm_margin"
output_path = os.path.join(output_dir, f"{file_name}_boxplot_comparison.png")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load JSON data
with open(input_path, "r") as f:
    data = json.load(f)

# Dynamically collect model values
model_data = defaultdict(list)

for image_index, models_values in data.items():
    if image_index == "metadata":
        continue
    for model_name, values in models_values.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Skipping invalid values for {model_name} at image {image_index}")
        model_data[model_name].append(values[0])

# Sort model names for consistent order
sorted_model_names = sorted(model_data.keys())
data_to_plot = [model_data[name] for name in sorted_model_names]

# Generate distinct colors
num_models = len(sorted_model_names)
colors = plt.cm.tab10(np.linspace(0, 1, num_models))

# Create and configure plot
plt.figure(figsize=(10, 7))
bp = plt.boxplot(data_to_plot,
                 patch_artist=True,
                 vert=True,
                 labels=sorted_model_names,
                 medianprops={"color": "black", "linewidth": 1.5})

# Customize colors
for patch, color in zip(bp["boxes"], colors):
    patch.set(facecolor=color, alpha=0.6)

# Add title and labels
plt.title(f"Model Performance Comparison for {data['metadata']['kernel_type']} kernel", fontsize=14)
plt.ylabel("Values", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

# Save and show plot
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()