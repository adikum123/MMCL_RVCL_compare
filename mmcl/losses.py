import diffdist
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from mmcl.solvers import *


def compute_kernel(X, Y, gamma=0.1, kernel_type="rbf", degree=3.0):
    if kernel_type == "linear":
        kernel = torch.mm(X, Y.T)
    elif kernel_type == "rbf":
        if gamma == "auto":
            gamma = 1.0 / X.shape[-1]
        else:
            gamma = float(gamma)
        X_norm = torch.sum(X**2, dim=1, keepdim=True)
        Y_norm = torch.sum(Y**2, dim=1, keepdim=True).T
        distances = X_norm + Y_norm - 2 * torch.mm(X, Y.T)
        kernel = torch.exp(-gamma * distances)
    elif kernel_type == "poly":
        kernel = torch.pow(torch.mm(X, Y.T) + 0.5, degree)
    elif kernel_type == "tanh":
        kernel = torch.tanh(gamma * torch.mm(X, Y.T))
    elif kernel_type == "min":
        # kernel = torch.minimum(torch.relu(X), torch.relu(Y))
        kernel = torch.min(
            torch.relu(X).unsqueeze(1), torch.relu(Y).unsqueeze(1).transpose(1, 0)
        ).sum(2)

    return kernel


class MMCL_PGD(nn.Module):
    def __init__(
        self,
        device,
        sigma=0.07,
        contrast_mode="all",
        base_sigma=0.07,
        batch_size=256,
        anchor_count=2,
        C=1.0,
        kernel="rbf",
        reg=0.1,
        schedule=[],
        multiplier=2,
        num_iters=1000,
        eta=1e-3,
        stop_condition=1e-2,
        solver_type="nesterov",
        use_norm="true",
    ):
        super(MMCL_PGD, self).__init__()
        self.sigma = sigma
        self.contrast_mode = contrast_mode
        self.base_sigma = base_sigma
        self.C = C
        self.kernel = kernel

        nn = batch_size - 1
        bs = batch_size
        self.device = device
        print(f"========> Using device in losses.py: {self.device}")
        self.schedule = schedule
        self.multiplier = multiplier
        self.anchor_count = anchor_count
        self.reg = reg

        self.num_iters = num_iters
        self.eta = eta
        self.stop_condition = stop_condition
        self.solver_type = solver_type
        self.use_norm = False if use_norm == "false" else True

    def get_kmask(self, bs):
        KMASK = torch.ones(bs, bs, bs, device=self.device).bool()
        for t in range(bs):
            KMASK[t, t, :] = False
            KMASK[t, :, t] = False
        return KMASK.detach()

    def get_mask(self, batch_size, anchor_count):
        mask = torch.eye(batch_size, dtype=torch.float32, device=self.device)

        mask = mask.repeat(anchor_count, anchor_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask, device=self.device),
            1,
            torch.arange(batch_size * anchor_count, device=self.device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask
        return mask, logits_mask

    def forward(self, z):
        n = z.shape[0]
        assert n % self.multiplier == 0

        ftr = F.normalize(z, p=2, dim=1)

        # Directly set ftr to z since distributed logic is removed
        ftr = torch.cat(torch.unbind(ftr, dim=1), dim=0)

        bs = ftr.shape[0] // 2
        nn = bs - 1

        K = compute_kernel(
            ftr[: nn + 1], ftr, kernel_type=self.kernel, gamma=self.sigma
        )

        # create blocks
        block = torch.zeros(bs, 2 * bs, device=self.device).bool()
        block[:bs, :bs] = True
        block12 = torch.zeros(bs, 2 * bs, device=self.device).bool()
        block12[:bs, bs:] = True
        mask, logits_mask = self.get_mask(bs, self.anchor_count)
        eye = torch.eye(self.anchor_count * bs, device=self.device)

        pos_mask = mask[:bs, bs:].bool()
        neg_mask = (mask * logits_mask + 1) % 2
        neg_mask = neg_mask - eye
        neg_mask = neg_mask[:bs, bs:].bool()

        kmask = torch.ones(bs, device=self.device).bool()
        kmask.requires_grad = False

        oneone = (
            torch.ones(bs, bs, device=self.device)
            + torch.eye(bs, device=self.device) * self.reg
        )
        one_bs = torch.ones(bs, nn, 1, device=self.device)
        one = torch.ones(nn, device=self.device)
        KMASK = self.get_kmask(bs)
        no_diag = (1 - torch.eye(bs, device=self.device)).bool()

        with torch.no_grad():
            KK = torch.masked_select(K.detach(), block).reshape(bs, bs)

            KK_d0 = KK * no_diag
            KXY = -KK_d0.unsqueeze(1).repeat(1, bs, 1)
            KXY = KXY + KXY.transpose(2, 1)
            Delta = (oneone + KK).unsqueeze(0) + KXY

            DD = torch.masked_select(Delta, KMASK).reshape(bs, nn, nn)

            if self.C == -1:
                alpha_y = torch.relu(torch.randn(bs, nn, 1, device=DD.device))
            else:
                alpha_y = torch.relu(torch.randn(bs, nn, 1, device=DD.device)).clamp(
                    min=0, max=self.C
                )

            if self.solver_type == "nesterov":
                alpha_y, iter_no, abs_rel_change, rel_change_init = pgd_with_nesterov(
                    self.eta,
                    self.num_iters,
                    DD,
                    2 * one_bs,
                    alpha_y.clone(),
                    self.C,
                    use_norm=self.use_norm,
                    stop_condition=self.stop_condition,
                )
            elif self.solver_type == "vanilla":
                alpha_y, iter_no, abs_rel_change, rel_change_init = pgd_simple_short(
                    self.eta,
                    self.num_iters,
                    DD,
                    2 * one_bs,
                    alpha_y.clone(),
                    self.C,
                    use_norm=self.use_norm,
                    stop_condition=self.stop_condition,
                )

            alpha_y = alpha_y.squeeze(2)
            if self.C == -1:
                alpha_y = torch.relu(alpha_y)
            else:
                alpha_y = torch.relu(alpha_y).clamp(min=0, max=self.C).detach()
            alpha_x = alpha_y.sum(1)

        Ks = torch.masked_select(K, block12).reshape(bs, bs)
        Kn = torch.masked_select(Ks.T, neg_mask).reshape(bs, nn).T

        pos_loss = (alpha_x * (Ks * pos_mask).sum(1)).mean()
        neg_loss = (alpha_y.T * Kn).sum() / bs
        loss = neg_loss - pos_loss

        sparsity = (alpha_y == self.C).sum() / ((alpha_y > 0).sum() + 1e-10)
        num_zero = (alpha_y == 0).sum() / alpha_y.numel()
        return (
            loss,
            (Ks * pos_mask).sum(1).mean(),
            Kn.mean(),
            sparsity,
            num_zero,
            0.0,
        )
