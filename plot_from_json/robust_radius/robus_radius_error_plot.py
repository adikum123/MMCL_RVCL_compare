import json
import math
import os

import matplotlib.pyplot as plt

# Load data
file_name = "mmcl_rbf-adversarial_cl-cl_info_nce"
with open(f"radius_results/{file_name}.json", "r") as f:
    data = json.load(f)

# Initialize lists
mmcl_means, regular_cl_means, rvcl_means = [], [], []
mmcl_ste, regular_cl_ste, rvcl_ste = [], [], []

# Process data with sorted image indices
sorted_indices = sorted(data.keys(), key=lambda x: int(x))  # Ensure proper order

for image_index in sorted_indices:
    models_values = data[image_index]
    for model_name, values in models_values.items():
        mean_val = values[0]
        std_val = values[1]
        ste_val = std_val / math.sqrt(3)  # Calculate standard error

        if model_name == "mmcl":
            mmcl_means.append(mean_val)
            mmcl_ste.append(ste_val)
        elif model_name == "regular_cl":
            regular_cl_means.append(mean_val)
            regular_cl_ste.append(ste_val)
        elif model_name == "rvcl":
            rvcl_means.append(mean_val)
            rvcl_ste.append(ste_val)
        else:
            raise ValueError(f"Unknown model: {model_name}")

# Create plot
plt.figure(figsize=(12, 6))
x = range(len(sorted_indices))  # X-axis as index positions

# Plot error bars for all models
plt.errorbar(x, mmcl_means, yerr=mmcl_ste,
             fmt="o-", capsize=4, label="MMCL")
plt.errorbar(x, regular_cl_means, yerr=regular_cl_ste,
             fmt="s--", capsize=4, label="Regular CL")
plt.errorbar(x, rvcl_means, yerr=rvcl_ste,
             fmt="^-.", capsize=4, label="Adversarial CL")

# Formatting
plt.xticks(x, sorted_indices)  # Show actual image indices on x-axis
plt.xlabel("Image Index")
plt.ylabel("Mean Value")
plt.title("Model Comparison with Standard Error (n=3)")
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join("plots/robust_radius", f"{file_name}_error_plot.png"), dpi=300)
plt.show()