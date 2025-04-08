import json

import matplotlib.pyplot as plt
import numpy as np

# Load the JSON file
with open('radius_results/mmcl_cnn_4layer_b_rvcl_cnn_4layer_b_adv_regular_cl_cnn_4layer_b.json', 'r') as f:
    radius_results = json.load(f)

# Extract data
mmcl_means = [radius_results[key]['mmcl'][0] for key in radius_results]
mmcl_stds = [radius_results[key]['mmcl'][1] for key in radius_results]
rvcl_means = [radius_results[key]['rvcl'][0] for key in radius_results]
rvcl_stds = [radius_results[key]['rvcl'][1] for key in radius_results]
regular_cl_means = [radius_results[key]['regular_cl'][0] for key in radius_results]
regular_cl_stds = [radius_results[key]['regular_cl'][1] for key in radius_results]

# Combine into lists for plotting
data = [mmcl_means, rvcl_means, regular_cl_means]
labels = ['MMCL', 'RVCL', 'Regular CL']
colors = ['blue', 'green', 'red']

# Create a box plot
plt.figure(figsize=(8, 6))
bp = plt.boxplot(data, patch_artist=True, vert=True, labels=labels)

# Set colors
for patch, color in zip(bp['boxes'], colors):
    patch.set(facecolor=color, alpha=0.5)

# Overlay scatter plot with error bars
for i, (means, stds, color) in enumerate(zip([mmcl_means, rvcl_means, regular_cl_means],
                                             [mmcl_stds, rvcl_stds, regular_cl_stds],
                                             colors)):
    x = np.random.normal(i + 1, 0.04, size=len(means))  # Jitter for better visibility
    plt.errorbar(x, means, yerr=stds, fmt='o', color=color, alpha=0.6,
                 ecolor='black', capsize=3, markersize=5)

# Labels and title
plt.xlabel('Model')
plt.ylabel('Robust Radius')
plt.title('Comparison of Robust Radius Across Models')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the new plot
output_path = 'plots/robust_radius/robust_radius_comparison_boxplot.png'
plt.savefig(output_path)
plt.show()