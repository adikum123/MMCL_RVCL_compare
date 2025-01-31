import copy
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import mmcl.utils as utils
from beta_crown.auto_LiRPA import BoundedModule, BoundedTensor
from beta_crown.auto_LiRPA.perturbations import *
from beta_crown.model_beta_CROWN import LiRPAConvNet, return_modify_model
from beta_crown.relu_conv_parallel import relu_bab_parallel
from beta_crown.utils import *
from rocl.attack_lib import FastGradientSignUntargeted, RepresentationAdv


class RobustRadius:

    def __init__(self, hparams, model_type=None):
        assert model_type in ['mmcl', 'rvcl']
        self.args = hparams
        self.device = torch.device('cpu')
        # Model
        print('==> Building model..')
        self.model_ori = utils.load_model_contrastive_test(
            model=self.args.mmcl_model if model_type=='mmcl' else self.args.rvcl_model,
            model_path=self.args.mmcl_checkpoint if model_type=='mmcl' else self.args.rvcl_load_checkpoint,
            device=self.device
        )
        print(f"Built model: {self.model_ori}")
        self.output_size = list(self.model_ori.children())[-1].weight.data.shape[0]
        self.img_clip = min_max_value(self.args)
        self.upper_eps = (torch.max(self.img_clip['max']) - torch.min(self.img_clip['min'])).item()



    def generate_attack(args, ori, target):
        pgd_target = RepresentationAdv(
            self.model_ori,
            None,
            epsilon=self.args.target_eps,
            alpha=self.args.alpha,
            min_val=img_clip['min'].to(self.device),
            max_val=img_clip['max'].to(self.device),
            max_iters=self.args.k,
            _type=self.args.attack_type,
            loss_type=self.args.loss_type
        )
        adv_target = pgd_target.attack_pgd(original_images=ori.to(self.device), target=target.to(self.device), type='attack', random_start=True)

        pgd = RepresentationAdv(
            self.model_ori,
            None,
            epsilon=self.args.epsilon,
            alpha=self.args.alpha,
            min_val=img_clip['min'].to(self.device),
            max_val=img_clip['max'].to(self.device),
            max_iters=self.args.k,
            _type=self.args.attack_type,
            loss_type=self.args.loss_type
        )
        adv_img = pgd.attack_pgd(original_images=ori.to(self.device), target=adv_target.to(self.device), type='sim', random_start=True)
        return adv_target, adv_img

    def unsupervised_search(
        self,
        model_ori,
        data,
        ori,
        target,
        norm,
        args,
        output_size,
        data_max=None,
        data_min=None,
        upper = 1.0,
        lower = 0.0,
        tol = 0.000001,
        max_steps = 100
    ):
        model = return_modify_model(model_ori, ori, target, output_size, contrastive=True, simplify=True)
        model.to(self.device)
        bound_modify_net = BoundedModule(model, torch.empty_like(data.to(self.device)), bound_opts={"conv_mode": "patches"}, device=self.device)

        step = 0
        while upper-lower > tol:
            eps = 0.5 * (lower + upper)
            if norm == np.inf:
                if data_max is None:
                    data_ub = data + eps  # torch.min(data + eps, data_max)  # eps is already normalized
                    data_lb = data - eps  # torch.max(data - eps, data_min)
                else:
                    data_ub = torch.min(data + eps, data_max)
                    data_lb = torch.max(data - eps, data_min)
            else:
                data_ub = data_lb = data
            ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb.to(self.device), x_U=data_ub.to(self.device))
            image = BoundedTensor(data, ptb).to(self.device)
            # indicates mode
            lb, _ = copy.deepcopy(bound_modify_net).compute_bounds(x=(image,), method='backward')
            lb = lb.item()
            print("[binary search] step = {}, current = {:.6f}, success = {}, val = {:.2f}".format(step,eps,lb > 0,lb))

            if lb > 0: # success at current value
                lower = eps
            else:
                upper = eps
            step += 1
            if step >= max_steps:
                break
        return lower, step, model

    def verify(self, img_ori, img_target):
        img_ori = img_ori.unsqueeze(0).to(self.device)
        img_target = img_target.unsqueeze(0).to(self.device)
        # normalize inputs
        ori_encoding = self.model_ori(img_ori.detach())
        target_encoding = self.model_ori(img_target.detach())
        # normalize
        f_ori = F.normalize(ori_encoding, p=2, dim=1)
        f_target = F.normalize(target_encoding, p=2, dim=1)
        # find lower bound
        verifier_lower, _, _ = self.unsupervised_search(
            self.model_ori,
            img_ori,
            f_ori,
            f_target,
            self.args.norm,
            self.args,
            self.output_size,
            self.img_clip['max'],
            self.img_clip['min'],
            upper=self.upper_eps,
            lower=0.0,
            max_steps=self.args.max_steps
        )
        return verifier_lower