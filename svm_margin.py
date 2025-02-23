import argparse
import math

import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm import SVC


def compute_margin(positive, negatives, args):
    if args.normalize:
        positive = F.normalize(positive, p=2, dim=-1)
        negatives = [F.normalize(x, p=2, dim=-1) for x in negatives]
    # Stack positive and negative samples
    X = np.vstack(
        [positive.detach().cpu().numpy()] + [neg.detach().cpu().numpy() for neg in negatives]
    )
    # Create labels (1 for positive, -1 for negatives)
    Y = np.hstack((np.ones(1), -np.ones(len(negatives))))

    # Extract SVM parameters from args
    svm_params = {
        "C": args.C,
        "kernel": args.kernel_type,
        "gamma": getattr(args, "kernel_gamma", "auto"),
        "degree": int(getattr(args, "degree", 3)),
        "coef0": getattr(args, "coef0", 0.0),
    }

    # Train SVM
    model = SVC(**svm_params)
    model.fit(X, Y)

    # Extract support vectors and dual coefficients
    support_vectors = model.support_vectors_
    dual_coefs = model.dual_coef_[0]

    # Compute kernel parameters
    kernel_type = svm_params["kernel"]
    if kernel_type in {"rbf", "poly"}:
        if svm_params["gamma"] == "auto":
            svm_params["gamma"] = 1 / X.shape[1]
        else:
            svm_params["gamma"] = float(svm_params["gamma"])
        kernel_params = {"gamma": svm_params["gamma"]}
    elif kernel_type == "poly" or kernel_type == "sigmoid":
        kernel_params = {
            "gamma": svm_params["gamma"],
            "degree": svm_params["degree"],
            "coef0": svm_params["coef0"],
        }
    elif kernel_type == "linear":
        kernel_params = {}
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")
    # Compute kernel matrix
    kernel_matrix = pairwise_kernels(X=support_vectors, metric=kernel_type, **kernel_params)
    # Compute ||w||^2
    w_norm_sq = np.dot(np.dot(dual_coefs, kernel_matrix), dual_coefs.T)
    # Return the margin (2 / ||w||)
    return 2 / math.sqrt(w_norm_sq) if w_norm_sq > 0 else w_norm_sq