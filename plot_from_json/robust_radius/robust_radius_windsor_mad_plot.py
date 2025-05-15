import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_abs_deviation as mad

# Load data
file_name = "mmcl_rbf-adversarial_cl-cl_info_nce"
with open(f"radius_results/{file_name}.json", "r") as f:
    data = json.load(f)

# Initialize dictionaries to store robust stats
models = ["mmcl", "adversarial_cl", "regular_cl"]
model_data = {model: {"means": []} for model in models}

# Collect all means first
for image_index in data:
    for model in models:
        if model == "adversarial_cl":
            model_data[model]["means"].append(data[image_index]["rvcl"][0])
            continue
        model_data[model]["means"].append(data[image_index][model][0])


# Calculate robust statistics for each model
robust_stats = {}
for model in models:
    means = np.array(model_data[model]["means"])
    median = np.median(means)
    mad_value = mad(means)

    # Winsorize means (clip extreme values)
    lower_bound = np.percentile(means, 5)
    upper_bound = np.percentile(means, 95)
    winsorized = np.clip(means, lower_bound, upper_bound)

    robust_stats[model] = {
        "median": median,
        "mad": mad_value,
        "winsorized_mean": np.mean(winsorized),
        "winsorized_std": np.std(winsorized)
    }

# Create comparison plot
plt.figure(figsize=(10, 6))

# Plot both median + MAD and winsorized mean ± std
x = np.arange(len(models))
width = 0.4

# Median with MAD error bars
plt.errorbar(x - width/2,
             [robust_stats[m]["median"] for m in models],
             yerr=[robust_stats[m]["mad"] for m in models],
             fmt="o", capsize=5, label="Median ± MAD")

# Winsorized mean with std error bars
plt.errorbar(x + width/2,
             [robust_stats[m]["winsorized_mean"] for m in models],
             yerr=[robust_stats[m]["winsorized_std"] for m in models],
             fmt="s", capsize=5, label="Winsorized Mean ± Std")

plt.xticks(x, [m.upper() for m in models])
plt.ylabel("Value")
plt.title("Robust Model Comparison (Outlier-Resistant)")
plt.legend(loc="upper left")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join("plots/robust_radius", f"{file_name}_windsor_mad.png"), dpi=300)
plt.show()