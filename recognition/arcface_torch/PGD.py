import os

import torch
import torch.nn.functional as F
import torch.utils.data.distributed

import losses
from partial_fc import PartialFC
from utils.utils_config import get_config


class PGDAttacker:
    def __init__(self, radius, steps, step_size, random_start, norm_type, args):
        self.radius = radius / 255.
        self.steps = steps
        self.step_size = step_size / 255.
        self.random_start = random_start
        self.norm_type = norm_type
        self.args = args

    def perturb(self, backbone, x, label):
        cfg = get_config(self.args.config)

        try:
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
        except KeyError:
            world_size = 1
            rank = 0

        local_rank = self.args.local_rank
        torch.cuda.set_device('cuda:' + str(local_rank))

        margin_softmax = losses.get_loss(cfg.loss)
        module_partial_fc = PartialFC(
            rank=rank, local_rank=local_rank, world_size=world_size, resume=cfg.resume,
            batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
            sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output)

        opt_backbone = torch.optim.SGD(
            params=[{'params': backbone.parameters()}],
            lr=cfg.lr / 512 * cfg.batch_size * world_size,
            momentum=0.9, weight_decay=cfg.weight_decay)
        opt_pfc = torch.optim.SGD(
            params=[{'params': module_partial_fc.parameters()}],
            lr=cfg.lr / 512 * cfg.batch_size * world_size,
            momentum=0.9, weight_decay=cfg.weight_decay)

        if self.steps == 0 or self.radius == 0:
            return x.clone()
        adv_x = x.clone()
        if self.random_start:
            if self.norm_type == 'l-infty':
                adv_x += (torch.rand_like(x) - 0.5) * self.radius
            else:
                adv_x += (torch.rand_like(x) - 0.5) * self.radius / self.steps
            self._clip_(adv_x, x)

        backbone.eval()
        for step in range(self.steps):
            adv_x.requires_grad_()
            features = F.normalize(backbone(adv_x))
            x_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc)

            # assert (features.grad == True)
            tem = x_grad.data * features
            looo = tem.sum() / 32  # batch size
            grad = torch.autograd.grad(looo, [adv_x])[0]
            # grad = torch.autograd.grad(tem, [adv_x],grad_outputs=torch.ones_like(tem))[0]

            with torch.no_grad():
                if self.norm_type == 'l-infty':
                    adv_x.add_(torch.sign(grad), alpha=self.step_size)
                else:
                    if self.norm_type == 'l2':
                        grad_norm = (grad.reshape(grad.shape[0], -1) ** 2).sum(dim=1).sqrt()
                    elif self.norm_type == 'l1':
                        grad_norm = grad.reshape(grad.shape[0], -1).abs().sum(dim=1)
                    grad_norm = grad_norm.reshape(-1, *([1] * (len(x.shape) - 1)))
                    scaled_grad = grad / (grad_norm + 1e-10)
                    adv_x.add_(scaled_grad, alpha=self.step_size)

                self._clip_(adv_x, x)

        return adv_x.data

    def _clip_(self, adv_x, x):
        adv_x -= x
        if self.norm_type == 'l-infty':
            adv_x.clamp_(-self.radius, self.radius)
        else:
            if self.norm_type == 'l2':
                norm = (adv_x.reshape(adv_x.shape[0], -1) ** 2).sum(dim=1).sqrt()
            elif self.norm_type == 'l1':
                norm = adv_x.reshape(adv_x.shape[0], -1).abs().sum(dim=1)
            norm = norm.reshape(-1, *([1] * (len(x.shape) - 1)))
            adv_x /= (norm + 1e-10)
            adv_x *= norm.clamp(max=self.radius)
        adv_x += x
        adv_x.clamp_(-0.5, 0.5)
