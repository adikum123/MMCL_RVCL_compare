import json
import os
from collections import defaultdict

import pandas as pd

file_name = "mmcl_cnn_4layer_b_rvcl_cnn_4layer_b_adv_regular_cl_cnn_4layer_b_supervised_cnn_4layer_b"
with open(f"../rs_results/{file_name}.json", "r") as f:
    data = json.load(f)

sigma_values = [0.25, 0.5, 1]
certified_radius_choices = [0, 0.5, 1, 1.5, 2, 2.5, 3]
model_names = ["mmcl", "rvcl", "regular_cl", "supervised"]
per_model = defaultdict(list)
per_sigma_radius = defaultdict(list)
for model in model_names:
    for curr_sigma in sigma_values:
        for curr_radius in certified_radius_choices:
            curr_values = [x for x in data[model] if x["sigma"] == curr_sigma and x["radius"] >= curr_radius]
            certified_instance_accuracy = sum(1 for x in curr_values if x["true_label"] == x["rs_label"]) / len(curr_values) if len(curr_values) > 0 else 0
            unchanged_percentage = sum(1 for x in curr_values if x["predicted_label"] == x["rs_label"]) / len(curr_values) if len(curr_values) > 0 else 0
            per_model[model].append({
                "sigma": curr_sigma,
                "radius": curr_radius,
                "certified_instance_accuracy": certified_instance_accuracy,
                "unchanged_percentage": unchanged_percentage
            })
            per_sigma_radius[f"sigma:{curr_sigma}|radius:{curr_radius}"].append({
                "model": model,
                "certified_instance_accuracy": certified_instance_accuracy,
                "unchanged_percentage": unchanged_percentage
            })

per_sigma_radius_updated = {}
for key, values in per_sigma_radius.items():
    max_certified_instance_accuracy = max(x["certified_instance_accuracy"] for x in values)
    best_certified_instance_accuracy_model = (
        [x["model"] for x in values if x["certified_instance_accuracy"] == max_certified_instance_accuracy]
        if max_certified_instance_accuracy > 0
        else []
    )
    max_unchanged_percentage = max(x["unchanged_percentage"] for x in values)
    best_unchanged_percentage_model = (
        [x["model"] for x in values if x["unchanged_percentage"] == max_unchanged_percentage]
        if max_unchanged_percentage > 0
        else []
    )
    per_sigma_radius[key] = {
        "per_model_values": values,
        "best_certified_instance_accuracy_models": best_certified_instance_accuracy_model,
        "best_unchanged_percentage_models": best_unchanged_percentage_model
    }

print(json.dumps(per_sigma_radius, indent=4))