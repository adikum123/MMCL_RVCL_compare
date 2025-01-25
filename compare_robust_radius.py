import argparse
import copy
import gc
import json
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch.nn.functional as F

import rocl.data_loader as data_loader
from beta_crown.auto_LiRPA import BoundedModule, BoundedTensor
from beta_crown.auto_LiRPA.perturbations import *
from beta_crown.model_beta_CROWN import LiRPAConvNet, return_modify_model
from beta_crown.relu_conv_parallel import relu_bab_parallel
from beta_crown.utils import *
from rocl.attack_lib import FastGradientSignUntargeted, RepresentationAdv

parser = argparse.ArgumentParser(description="unsupervised beta binary search")

##### arguments for beta CROWN #####
parser.add_argument(
    "--no_solve_slope",
    action="store_false",
    dest="solve_slope",
    help="do not optimize slope/alpha in compute bounds",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    choices=["cpu", "cuda"],
    help="use cpu or cuda",
)
parser.add_argument("--gpuno", default="0", type=str)

parser.add_argument(
    "--norm", type=float, default="inf", help="p norm for epsilon perturbation"
)
parser.add_argument(
    "--bound_type",
    type=str,
    default="CROWN-IBP",
    choices=["IBP", "CROWN-IBP", "CROWN"],
    help="method of bound analysis",
)
parser.add_argument(
    "--mmcl_model",
    type=str,
    default="cnn_4layer_b",
    help="model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)",
)
parser.add_argument(
    "--rvcl_model",
    type=str,
    default="cnn_4layer_b",
    help="model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)",
)
parser.add_argument(
    "--bound_opts",
    type=str,
    default="same-slope",
    choices=["same-slope", "zero-lb", "one-lb"],
    help="bound options for relu",
)
parser.add_argument(
    "--no_warm",
    action="store_true",
    default=False,
    help="using warm up for lp solver, true by default",
)
parser.add_argument(
    "--no_beta",
    action="store_true",
    default=False,
    help="using beta splits, true by default",
)
parser.add_argument(
    "--max_subproblems_list",
    type=int,
    default=200000,
    help="max length of sub-problems list",
)
parser.add_argument(
    "--decision_thresh",
    type=float,
    default=0,
    help="decision threshold of lower bounds",
)
parser.add_argument(
    "--timeout", type=float, default=30, help="timeout for one property"
)
parser.add_argument(
    "--mode",
    type=str,
    default="incomplete",
    choices=["complete", "incomplete", "verified-acc"],
    help="which mode to use",
)

##### arguments for model #####
parser.add_argument(
    "--train_type",
    default="contrastive",
    type=str,
    help="contrastive/linear eval/test/supervised",
)
parser.add_argument("--dataset", default="cifar-10", type=str, help="cifar-10/mnist")
parser.add_argument(
    "--mmcl_load_checkpoint", default="", type=str, help="PATH TO CHECKPOINT"
)
parser.add_argument(
    "--rvcl_load_checkpoint", default="", type=str, help="PATH TO CHECKPOINT"
)
parser.add_argument("--name", default="", type=str, help="name of run")
parser.add_argument("--seed", default=1, type=int, help="random seed")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")

##### arguments for data augmentation #####
parser.add_argument(
    "--color_jitter_strength",
    default=0.5,
    type=float,
    help="0.5 for CIFAR, 1.0 for ImageNet",
)
parser.add_argument(
    "--temperature", default=0.5, type=float, help="temperature for pairwise-similarity"
)

##### arguments for PGD attack & Adversarial Training #####
parser.add_argument("--attack_type", type=str, default="linf", help="adversarial l_p")
parser.add_argument(
    "--target_eps",
    type=float,
    default=16.0 / 255,
    help="maximum perturbation of adversaries (8/255 0.0314 for cifar-10)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.001,
    help="movement multiplier per iteration when generating adversarial examples (2/255=0.00784)",
)
parser.add_argument(
    "--k",
    type=int,
    default=150,
    help="maximum iteration when generating adversarial examples",
)
parser.add_argument("--random_start", type=bool, default=True, help="True for PGD")
parser.add_argument(
    "--loss_type", type=str, default="mse", help="loss type for Rep: mse/sim/l1/cos"
)

##### arguments for binary_search #####
parser.add_argument("--mini_batch", type=int, default=10, help="mini batch for PGD")
parser.add_argument(
    "--ver_total", type=int, default=100, help="number of img to verify"
)
parser.add_argument("--max_steps", type=int, default=200, help="max steps for search")
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
img_clip = min_max_value(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading MMCL and RVCL models")
mmcl_model = torch.load(args.mmcl_load_checkpoint, map_location=device)
rvcl_model = torch.load(args.rvcl_load_checkpoint, map_location=device)


def generate_attack(args, model_ori, ori, target):
    pgd_target = RepresentationAdv(
        model_ori,
        None,
        epsilon=args.target_eps,
        alpha=args.alpha,
        min_val=img_clip["min"].to(args.device),
        max_val=img_clip["max"].to(args.device),
        max_iters=args.k,
        _type=args.attack_type,
        loss_type=args.loss_type,
    )
    adv_target = pgd_target.attack_pgd(
        original_images=ori.to(args.device),
        target=target.to(args.device),
        type="attack",
        random_start=True,
    )

    pgd = RepresentationAdv(
        model_ori,
        None,
        epsilon=args.epsilon,
        alpha=args.alpha,
        min_val=img_clip["min"].to(args.device),
        max_val=img_clip["max"].to(args.device),
        max_iters=args.k,
        _type=args.attack_type,
        loss_type=args.loss_type,
    )
    adv_img = pgd.attack_pgd(
        original_images=ori.to(args.device),
        target=adv_target.to(args.device),
        type="sim",
        random_start=True,
    )
    return adv_target, adv_img


def generate_ver_data(loader, model, total, class_num, adv=True):
    count = [0 for _ in range(class_num)]
    per_class = total // class_num
    data_loader = iter(loader)
    ans_image = []
    if adv:
        adv_target = []
        adv_eps = []
    ans_label = []
    while sum(count) < total:
        (ori, aug_img, _, label) = next(data_loader)
        i = int(label)
        if count[i] < per_class:
            ans_image.append(ori)
            ans_label.append(i)
            if adv:
                i1, i2 = generate_attack(args, model, ori, aug_img)
                adv_target.append(i1)
                adv_eps.append(i2)
            count[i] += 1
    if adv:
        return ans_image, adv_target, adv_eps, ans_label
    else:
        return ans_image, ans_label


def unsupervised_search(
    model_ori,
    data,
    ori,
    target,
    norm,
    args,
    output_size,
    data_max=None,
    data_min=None,
    upper=1.0,
    lower=0.0,
    tol=0.000001,
    max_steps=100,
):
    model = LiRPAConvNet(
        model_ori,
        ori,
        target,
        output_size=output_size,
        contrastive=True,
        simplify=True,
        solve_slope=args.solve_slope,
        device=args.device,
        in_size=data.shape,
    )
    step = 0
    while upper - lower > tol:
        eps = 0.5 * (lower + upper)
        if norm == np.inf:
            if data_max is None:
                data_ub = (
                    data + eps
                )  # torch.min(data + eps, data_max)  # eps is already normalized
                data_lb = data - eps  # torch.max(data - eps, data_min)
            else:
                data_ub = torch.min(data + eps, data_max)
                data_lb = torch.max(data - eps, data_min)
        else:
            data_ub = data_lb = data
        ptb = PerturbationLpNorm(
            norm=norm, eps=eps, x_L=data_lb.to(args.device), x_U=data_ub.to(args.device)
        )
        x = BoundedTensor(data, ptb).to(args.device)
        domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)
        with HiddenPrints():
            lb, _, _, _ = relu_bab_parallel(
                copy.deepcopy(model),
                domain,
                x,
                batch=args.batch_size,
                no_LP=True,
                decision_thresh=args.decision_thresh,
                beta=not args.no_beta,
                max_subproblems_list=args.max_subproblems_list,
                timeout=args.timeout,
            )
        if isinstance(lb, torch.Tensor):
            lb = lb.item()
        if lb > 500:
            continue
        print(
            "[binary search] step = {}, current = {:.6f}, success = {}, val = {:.2f}".format(
                step, eps, lb > 0, lb
            )
        )

        if lb > 0:  # success at current value
            lower = eps
        else:
            upper = eps
        step += 1
        if step >= max_steps:
            break
    return lower, step, model.modify_net


# Data
print("==> Preparing data..")
_, _, testloader, testdst = data_loader.get_dataset(args)
mmcl_image, mmcl_label = generate_ver_data(
    testloader, mmcl_model, args.ver_total, class_num=10, adv=False
)
rvcl_image, rvcl_label = generate_ver_data(
    testloader, rvcl_model, args.ver_total, class_num=10, adv=True
)
robust_radius = defaultdict(list)

line_iter = 0
upper_eps = (torch.max(img_clip["max"]) - torch.min(img_clip["min"])).item()
total_avg = []
total_time = []
for batch_iter in range(args.ver_total // args.mini_batch):
    for ori_iter in range(args.mini_batch):
        total_ori_iter = batch_iter * args.mini_batch + ori_iter
        print("verifying {}-th image".format(total_ori_iter))
        avg_veri_lower = []
        avg_time = []
        avg_steps = []
        for target_iter in range(args.mini_batch):
            if ori_iter == target_iter:
                continue
            total_target_iter = batch_iter * args.mini_batch + target_iter
            print(
                "verifying {}-th image, against {}-th image".format(
                    total_ori_iter, total_target_iter
                )
            )
            img_ori = image[total_ori_iter]
            img_target = image[total_target_iter]

            # mmcl step
            mmcl_item = {}
            mmcl_model.to("cpu")
            f_ori = F.normalize(mmcl_model(mmcl_model.to("cpu").detach()), p=2, dim=1)
            f_target = F.normalize(
                mmcl_model(mmcl_model.to("cpu").detach()), p=2, dim=1
            )

            start = time.time()
            (mmcl_item["veri_lower"], mmcl_item["steps"], modify_net) = (
                unsupervised_search(
                    mmcl_model,
                    img_ori,
                    f_ori,
                    f_target,
                    args.norm,
                    args,
                    output_size,
                    img_clip["max"],
                    img_clip["min"],
                    upper=upper_eps,
                    lower=0.0,
                    max_steps=args.max_steps,
                )
            )
            mmcl_item["time"] = time.time() - start
            modify_net.to(args.device)
            mmcl_item["value_upper"] = modify_net(img_ori.to(args.device)).item()
            mmcl_item["value_lower"] = modify_net(img_target.to(args.device)).item()

            mmcl_item["ori"] = total_ori_iter
            mmcl_item["target"] = total_target_iter

            # rvcl step
            rvcl_item = {}
            rvcl_model.to("cpu")
            f_ori = F.normalize(rvcl_model(mmcl_model.to("cpu").detach()), p=2, dim=1)
            f_target = F.normalize(
                rvcl_model(mmcl_model.to("cpu").detach()), p=2, dim=1
            )

            start = time.time()
            (rvcl_item["veri_lower"], rvcl_item["steps"], modify_net) = (
                unsupervised_search(
                    rvcl_model,
                    img_ori,
                    f_ori,
                    f_target,
                    args.norm,
                    args,
                    output_size,
                    img_clip["max"],
                    img_clip["min"],
                    upper=upper_eps,
                    lower=0.0,
                    max_steps=args.max_steps,
                )
            )
            rvcl_item["time"] = time.time() - start
            modify_net.to(args.device)
            rvcl_item["value_upper"] = modify_net(img_ori.to(args.device)).item()
            rvcl_item["value_lower"] = modify_net(img_target.to(args.device)).item()

            rvcl_item["ori"] = total_ori_iter
            rvcl_item["target"] = total_target_iter

            robust_radius[(total_ori_iter, total_target_iter)].append(
                {"mmcl_item": mmcl_item, "rvcl_item": rvcl_item}
            )

print(robust_radius)
