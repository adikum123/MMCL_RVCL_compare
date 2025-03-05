import json
from collections import defaultdict

import numpy as np

file_path = "radius_results/mmcl_cnn_4layer_b_C_1_rbf_auto_rvcl_cifar10_cnn_4layer_b_adv2_regular_cl_regular_cl_cnn_4layer_b_bs_32_lr_1e-3.json"

with open(file_path, "r") as f:
    average_robust_radius = json.load(f)

retries = defaultdict(list)

for key, value in average_robust_radius["average_robust_radius"].items():
    for x in value:
        retries[x["retry_num"]].append({
            "mmcl": x["mmcl"],
            "rvcl": x["rvcl"],
            "regular_cl": x["regular_cl"],
        })

print(json.dumps(retries, indent=4))

averages_per_retry = defaultdict(list)
for retry, retry_values in retries.items():
    averages_per_retry[retry].append({
        "mmcl": np.mean(list(x["mmcl"] for x in retry_values)),
        "rvcl": np.mean(list(x["rvcl"] for x in retry_values)),
        "regular_cl": np.mean(list(x["regular_cl"] for x in retry_values))
    })

print(json.dumps(averages_per_retry, indent=4))

per_model_extracted_values = defaultdict(list)
for retry, values in averages_per_retry.items():
    for item in values:
        for model, average_radius in item.items():
            per_model_extracted_values[model].append(average_radius)

per_model_mean_std = defaultdict(list)
for model, values in per_model_extracted_values.items():
    per_model_mean_std[model].append({
        "mean": np.mean(values),
        "std": np.std(values)
    })

print(per_model_mean_std)