import argparse
import math

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm import SVC


def compute_margin(positive, negatives, args):
    X = np.vstack((positive.detach().numpy(), np.array([neg.detach().numpy() for neg in negatives.squeeze(1)])))
    Y = np.vstack((np.ones((1, 1)), -np.ones((len(negatives), 1))))
    # Extract SVM parameters from args
    svm_params = {
        "C": args.C,
        "kernel": args.kernel_type,
        "gamma": getattr(args, "gamma", "scale"),  # Default to 'scale' if gamma not specified
        "degree": getattr(args, "degree", 3),     # Default to 3 if degree not specified
        "coef0": getattr(args, "coef0", 0.0)      # Default to 0.0 if coef0 not specified
    }
    # Train SVM using the extracted parameters
    model = SVC(**svm_params)
    model.fit(X, Y.ravel())
    # Extract support vectors and dual coefficients
    support_vectors = model.support_vectors_
    dual_coefs = model.dual_coef_[0]
    # Extract kernel parameters based on the kernel type
    kernel_type = svm_params["kernel"]
    if kernel_type == "rbf":
        # Compute numeric value for gamma if it is 'scale' or 'auto'
        if svm_params["gamma"] == "scale":
            svm_params["gamma"] = 1 / (X.shape[1] * X.var())  # scale: 1 / (n_features * variance of X)
        elif svm_params["gamma"] == "auto":
            svm_params["gamma"] = 1 / X.shape[1]  # auto: 1 / n_features
        else:
            svm_params["gamma"] = float(svm_params["gamma"])
        kernel_params = {"gamma": svm_params["gamma"]}
    elif kernel_type == "poly" or kernel_type == "sigmoid":
        kernel_params = {
            "gamma": svm_params["gamma"],
            "degree": svm_params["degree"],
            "coef0": svm_params["coef0"]
        }
    elif kernel_type == "linear":
        kernel_params = {}
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")
    print(f'SVM params: {svm_params} \nKernel params: {kernel_params}')
    # Compute the kernel matrix
    kernel_matrix = pairwise_kernels(
        X=support_vectors,
        metric=kernel_type,
        **kernel_params,
    )
    w_norm_sq = np.dot(np.dot(dual_coefs, kernel_matrix), dual_coefs.T)
    # Return the margin (2 / ||w||)
    return 2 / math.sqrt(w_norm_sq) if w_norm_sq >= 0 else w_norm_sq
