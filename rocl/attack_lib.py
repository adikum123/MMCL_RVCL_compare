import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .loss import NT_xent, pairwise_similarity


def project(x, original_x, epsilon, _type="linf"):

    if _type == "linf":
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)
    else:
        raise NotImplementedError

    return x


def tensor_clip(x, min_x, max_x):
    ans = torch.min(x, max_x)
    return torch.max(ans, min_x)


class FastGradientSignUntargeted:
    """
    Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
    with iterative grad sign updates
    """

    def __init__(
        self,
        model,
        linear,
        epsilon,
        alpha,
        min_val,
        max_val,
        max_iters,
        device,
        _type="linf",
    ):

        # Model
        self.model = model
        self.linear = linear
        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type
        self.device = device

    def perturb(
        self, original_images, labels, reduction4loss="mean", random_start=True
    ):
        # original_images: values are within self.min_val and self.max_val
        # The adversaries created from random close points to the original data
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon
            )
            rand_perturb = rand_perturb.to(self.device)
            x = original_images.clone() + rand_perturb
            x = tensor_clip(x, self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True

        self.model.set_eval()
        if not self.linear == "None":
            self.linear.eval()

        with torch.enable_grad():
            for _iter in range(self.max_iters):

                self.model.zero_grad()
                if not self.linear == "None":
                    self.linear.zero_grad()

                if self.linear == "None":
                    outputs = self.model(x)
                else:
                    outputs = self.linear(self.model(x))

                loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)

                grad_outputs = None
                grads = torch.autograd.grad(
                    loss,
                    x,
                    grad_outputs=grad_outputs,
                    only_inputs=True,
                    retain_graph=False,
                )[0]

                if self._type == "linf":
                    scaled_g = torch.sign(grads.data)

                x.data += self.alpha * scaled_g

                x = tensor_clip(x, self.min_val, self.max_val)
                x = project(x, original_images, self.epsilon, self._type)

        return x.detach()


class RepresentationAdv:

    def __init__(
        self,
        model,
        projector,
        epsilon,
        alpha,
        min_val,
        max_val,
        max_iters,
        _type="linf",
        loss_type="sim",
        regularize="original",
    ):

        # Model
        self.model = model
        self.projector = projector
        self.regularize = regularize
        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type
        # loss type
        self.loss_type = loss_type

    def get_loss(self, original_images, target, optimizer, weight, random_start=True):
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon
            )
            rand_perturb = rand_perturb.float().to(self.device)
            x = original_images.float().clone() + rand_perturb

            # x = torch.clamp(x,self.min_val, self.max_val)
            x = tensor_clip(x, self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True

        self.model.eval()
        self.projector.eval()
        batch_size = len(x)

        with torch.enable_grad():
            for _iter in range(self.max_iters):

                self.model.zero_grad()
                self.projector.zero_grad()

                if self.loss_type == "mse":
                    loss = F.mse_loss(
                        self.projector(self.model(x)),
                        self.projector(self.model(target)),
                    )
                elif self.loss_type == "sim":
                    inputs = torch.cat((x, target))
                    output = self.projector(self.model(inputs))
                    similarity, _ = pairwise_similarity(
                        output, temperature=0.5, multi_gpu=False, adv_type="None"
                    )
                    loss = NT_xent(similarity, "None")
                elif self.loss_type == "l1":
                    loss = F.l1_loss(
                        self.projector(self.model(x)),
                        self.projector(self.model(target)),
                    )
                elif self.loss_type == "cos":
                    loss = (
                        1
                        - F.cosine_similarity(
                            self.projector(self.model(x)),
                            self.projector(self.model(target)),
                        ).mean()
                    )

                grads = torch.autograd.grad(
                    loss, x, grad_outputs=None, only_inputs=True, retain_graph=False
                )[0]

                if self._type == "linf":
                    scaled_g = torch.sign(grads.data)

                x.data += self.alpha * scaled_g

                x = tensor_clip(x, self.min_val, self.max_val)
                x = project(x, original_images, self.epsilon, self._type)

        self.model.train()
        self.projector.train()
        optimizer.zero_grad()

        if self.loss_type == "mse":
            loss = F.mse_loss(
                self.projector(self.model(x)), self.projector(self.model(target))
            ) * (1.0 / batch_size)
        elif self.loss_type == "sim":
            if self.regularize == "original":
                inputs = torch.cat((x, original_images))
            else:
                inputs = torch.cat((x, target))
            output = self.projector(self.model(inputs))
            similarity, _ = pairwise_similarity(
                output, temperature=0.5, multi_gpu=False, adv_type="None"
            )
            loss = (1.0 / weight) * NT_xent(similarity, "None")
        elif self.loss_type == "l1":
            loss = F.l1_loss(
                self.projector(self.model(x)), self.projector(self.model(target))
            ) * (1.0 / batch_size)
        elif self.loss_type == "cos":
            loss = 1 - F.cosine_similarity(
                self.projector(self.model(x)), self.projector(self.model(target))
            ).sum() * (1.0 / batch_size)

        return x.detach(), loss

    def attack_pgd(self, original_images, target, type="sim", random_start=True):
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon
            )
            rand_perturb = rand_perturb.float().to(self.device)
            x = original_images.float().clone() + rand_perturb

            # x = torch.clamp(x,self.min_val, self.max_val)
            x = tensor_clip(x, self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True

        self.model.eval()

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                self.model.zero_grad()

                if self.loss_type == "mse":
                    loss = F.mse_loss(self.model(x), self.model(target))
                elif self.loss_type == "sim":
                    inputs = torch.cat((x, target))
                    output = self.model(inputs)
                    similarity, _ = pairwise_similarity(
                        output, temperature=0.5, multi_gpu=False, adv_type="None"
                    )
                    loss = NT_xent(similarity, "None")
                elif self.loss_type == "l1":
                    loss = F.l1_loss(self.model(x), self.model(target))
                elif self.loss_type == "cos":
                    loss = (
                        1
                        - F.cosine_similarity(self.model(x), self.model(target)).mean()
                    )

                grads = torch.autograd.grad(
                    loss, x, grad_outputs=None, only_inputs=True, retain_graph=False
                )[0]
                # grads = torch.autograd.grad(loss, x)[0]

                if self._type == "linf":
                    scaled_g = torch.sign(grads.data)

                if type == "sim":
                    x.data -= self.alpha * scaled_g
                elif type == "attack":
                    x.data += self.alpha * scaled_g
                else:
                    raise AssertionError("no such attack type")
                x.data = tensor_clip(x.data, self.min_val, self.max_val)
                x.data = project(x.data, original_images, self.epsilon, self._type)

        return x.detach()

    def attack_gradient(self, original_images, target, random_start=True):
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon
            )
            rand_perturb = rand_perturb.float().to(self.device)
            x = original_images.float().clone() + rand_perturb
            x = tensor_clip(x, self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x = Variable(x)
        x.requires_grad = True
        optimizer = torch.optim.Adam([x], lr=0.01)
        self.model.eval()

        for _iter in range(self.max_iters):
            optimizer.zero_grad()
            loss = F.mse_loss(self.model(x), self.model(target))
            loss.backward()
            optimizer.step()
            x.data = tensor_clip(x.data, self.min_val, self.max_val)
            x.data = project(x.data, original_images, self.epsilon, self._type)

        return x.detach()
