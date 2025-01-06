from __future__ import print_function

import time

import numpy as np
import torch
import torch.autograd
import torch.nn as nn
from solvers import *
from torch.autograd import Variable


def compute_kernel(X, Y, kernel_type, gamma=0.1):
    if kernel_type == "linear":
        kernel = torch.mm(X, Y.T)
    elif kernel_type == "rbf":
        if gamma == "auto":
            gamma = 1 / X.shape[-1]
        gamma = 1.0 / float(gamma)
        # distances = torch.cdist(X, Y)
        distances = -gamma * (2 - 2.0 * torch.mm(X, Y.T))
        kernel = torch.exp(distances)

    elif kernel_type == "poly":
        kernel = torch.pow(torch.mm(X, Y.T) + 0.5, 3.0)
    elif kernel_type == "tanh":
        kernel = torch.tanh(gamma * torch.mm(X, Y.T))
    elif kernel_type == "min":
        # kernel = torch.minimum(torch.relu(X), torch.relu(Y))
        kernel = torch.min(
            torch.relu(X).unsqueeze(1), torch.relu(Y).unsqueeze(1).transpose(1, 0)
        ).sum(2)

    return kernel

class MMCL_inv(nn.Module):
    def __init__(self, kernel_type, sigma=0.07, batch_size=256, anchor_count=2, C=1.0, device=None):
        super(MMCL_inv, self).__init__()
        self.device = torch.device('cuda' if torch.cuda().is_available() else 'cpu') if device is None else device
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.C = C

        nn = batch_size - 1
        bs = batch_size
        self.mask, self.logits_mask = self.get_mask(batch_size, anchor_count)
        self.eye = torch.eye(anchor_count*batch_size).to(self.device)

        self.pos_mask = self.mask[:bs, bs:].bool()
        neg_mask=(self.mask*self.logits_mask+1)%2;
        self.neg_mask = neg_mask-self.eye
        self.neg_mask = self.neg_mask[:bs, bs:].bool()

        self.kmask = torch.ones(batch_size,).bool().to(self.device)
        self.kmask.requires_grad = False

        self.oneone = (torch.ones(bs, bs) + torch.eye(bs)*0.1).to(self.device)
        self.one_bs = torch.ones(batch_size, nn, 1).to(self.device)
        self.one = torch.ones(nn,).to(self.device)
        self.KMASK = self.get_kmask(bs)
        self.block = torch.zeros(bs,2*bs).bool().to(self.device); self.block[:bs,:bs] = True
        self.block12 = torch.zeros(bs,2*bs).bool().to(self.device); self.block12[:bs,bs:] = True
        self.no_diag = (1-torch.eye(bs)).bool().to(self.device)

    def get_kmask(self, bs):
        KMASK = torch.ones(bs, bs, bs).bool().to(self.device)
        for t in range(bs):
            KMASK[t,t,:] = False
            KMASK[t,:,t] = False
        return KMASK.detach()

    def get_mask(self, batch_size, anchor_count):
        mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)

        mask = mask.repeat(anchor_count, anchor_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask
        return mask, logits_mask

    def forward(self, features, labels=None, mask=None):
        bs = features.shape[0]
        nn = bs - 1

        F = torch.cat(torch.unbind(features, dim=1), dim=0)
        K = compute_kernel(F[:nn+1], F, gamma=self.sigma, type=self.hparams.kernel_type)


        with torch.no_grad():
            KK = torch.masked_select(K.detach(), self.block).reshape(bs, bs)

            KK_d0 = KK*self.no_diag

            KXY = -KK_d0.unsqueeze(1).repeat(1,bs,1)
            KXY = KXY + KXY.transpose(2,1)
            Delta = (self.oneone + KK).unsqueeze(0) + KXY

            DD = torch.masked_select(Delta, self.KMASK).reshape(bs, nn, nn)

            alpha_y, _ = torch.solve(2*self.one_bs, DD)
            alpha_y = alpha_y.squeeze(2)

            if self.C == -1:
                alpha_y = torch.relu(alpha_y).detach()
            else:
                alpha_y = torch.relu(alpha_y).clamp(min=0, max=self.C).detach()

            alpha_x = alpha_y.sum(1)

        Ks = torch.masked_select(K, self.block12).reshape(bs, bs)
        Kn = torch.masked_select(Ks.T, self.neg_mask).reshape(bs,nn).T

        pos_loss = (alpha_x*(Ks*self.pos_mask).sum(1)).mean()
        neg_loss = (alpha_y.T*Kn).sum()/bs
        loss = neg_loss - pos_loss

        return loss


class MMCL_pgd(nn.Module):
    def __init__(
        self,
        kernel_type,
        sigma=0.07,
        batch_size=256,
        anchor_count=2,
        C=1.0,
        num_iter=1000,
        eta=1E-3,
        stop_condition=0.01,
        solver_type='nesterov',
        use_norm='false',
        device=None
    ):
        super(MMCL_pgd, self).__init__()
        self.device = torch.device('cuda' if torch.cuda().is_available() else 'cpu') if device is None else device
        self.kernel_type=kernel_type
        self.sigma = sigma
        self.C = C

        nn = batch_size - 1
        bs = batch_size
        self.mask, self.logits_mask = self.get_mask(batch_size, anchor_count)
        self.eye = torch.eye(anchor_count*batch_size).to(self.device)

        self.pos_mask = self.mask[:bs, bs:].bool()
        neg_mask=(self.mask*self.logits_mask+1)%2
        self.neg_mask = neg_mask-self.eye
        self.neg_mask = self.neg_mask[:bs, bs:].bool()

        self.kmask = torch.ones(batch_size,).bool().to(self.device)
        self.kmask.requires_grad = False

        self.oneone = (torch.ones(bs, bs) + torch.eye(bs)*0.1).to(self.device)
        self.one_bs = torch.ones(batch_size, nn, 1).to(self.device)
        self.one = torch.ones(nn,).to(self.device)
        self.KMASK = self.get_kmask(bs)
        self.block = torch.zeros(bs,2*bs).bool().to(self.device); self.block[:bs,:bs] = True
        self.block12 = torch.zeros(bs,2*bs).bool().to(self.device); self.block12[:bs,bs:] = True
        self.no_diag = (1-torch.eye(bs)).bool().to(self.device)
        self.num_iters = num_iter
        self.eta = eta
        self.stop_condition = stop_condition
        self.solver_type = solver_type
        self.use_norm = False if use_norm=='false' else True


    def get_kmask(self, bs):
        KMASK = torch.ones(bs, bs, bs).bool().to(self.device)
        for t in range(bs):
            KMASK[t,t,:] = False
            KMASK[t,:,t] = False
        return KMASK.detach()

    def get_mask(self, batch_size, anchor_count):
        mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        mask = mask.repeat(anchor_count, anchor_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask
        return mask, logits_mask

    def forward(self, features, labels=None, mask=None):

        bs = features.shape[0]
        nn = bs - 1

        F = torch.cat(torch.unbind(features, dim=1), dim=0)
        K = compute_kernel(F[:nn+1], F, gamma=self.sigma, kernel_type=self.kernel_type)

        with torch.no_grad():
            KK = torch.masked_select(K.detach(), self.block).reshape(bs, bs)
            KK_d0 = KK*self.no_diag
            KK_d0[torch.isnan(KK_d0)] = 0.0
            KXY = -KK_d0.unsqueeze(1).repeat(1,bs,1)
            KXY = KXY + KXY.transpose(2,1)
            Delta = (self.oneone + KK).unsqueeze(0) + KXY
            DD = torch.masked_select(Delta, self.KMASK).reshape(bs, nn, nn)

            if self.C == -1:
                alpha_y = torch.relu(torch.randn(bs,nn,1,device=DD.device))
            else:
                alpha_y = torch.relu(torch.randn(bs,nn,1,device=DD.device)).clamp(min=0, max=self.C)

            if self.solver_type == 'nesterov':
                alpha_y,iter_no,abs_rel_change,rel_change_init = pgd_with_nesterov(
                    eta=self.eta,
                    num_iter=self.num_iters,
                    Q=DD,
                    p=1*self.one_bs,
                    alpha_y=alpha_y.clone(),
                    C=self.C,
                    use_norm=self.use_norm,
                    stop_condition=self.stop_condition
                )
            elif self.solver_type == 'vanilla':
                alpha_y,iter_no,abs_rel_change,rel_change_init = pgd_simple_short(
                    self.eta,
                    self.num_iters,
                    DD,
                    2*self.one_bs,
                    alpha_y.clone(),
                    self.C,
                    use_norm=self.use_norm,
                    stop_condition=self.stop_condition
                )

            alpha_y = alpha_y.squeeze(2)

            if self.C == -1:
                alpha_y = torch.relu(alpha_y).detach()
            else:
                alpha_y = torch.relu(alpha_y).clamp(min=0, max=self.C).detach()
            alpha_x = alpha_y.sum(1)

        Ks = torch.masked_select(K, self.block12).reshape(bs, bs)
        Kn = torch.masked_select(Ks.T, self.neg_mask).reshape(bs,nn).T
        pos_loss = (alpha_x*(Ks*self.pos_mask).sum(1)).mean()
        neg_loss = (alpha_y.T*Kn).sum()
        loss = neg_loss - pos_loss
        return loss


