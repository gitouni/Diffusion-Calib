import torch.optim.lr_scheduler as lr_scheduler
from typing import Literal

def get_lr_scheduler(optimizer,
        scheduler_type:Literal['step','mstep','exponential','cosine','cosine-warmup','poly'],
        **argv):
    if scheduler_type == 'step':
        return lr_scheduler.StepLR(optimizer, **argv)
    if scheduler_type == 'mstep':
        return lr_scheduler.MultiStepLR(optimizer, **argv)
    if scheduler_type == 'exponential':
        return lr_scheduler.ExponentialLR(optimizer, **argv)
    if scheduler_type == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, **argv)
    if scheduler_type == 'cosine-warmup':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **argv)
    if scheduler_type == 'poly':
        return lr_scheduler.PolynomialLR(optimizer, **argv)
    raise NotImplementedError("Unrecognized scheduler type:{}".format(scheduler_type))

