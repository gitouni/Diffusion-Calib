import torch.optim.lr_scheduler as lr_scheduler
from typing import Literal
from prodigyopt import Prodigy
from torch.optim import Adam, AdamW

def get_lr_scheduler(optimizer,
        scheduler_type:Literal['constant','step','mstep','exponential','cosine','cosine-warmup','poly'],
        **argv):
    scheduler_type = scheduler_type.lower()
    if scheduler_type == 'constant':
        return lr_scheduler.ConstantLR(optimizer, **argv)
    elif scheduler_type == 'step':
        return lr_scheduler.StepLR(optimizer, **argv)
    elif scheduler_type == 'mstep':
        return lr_scheduler.MultiStepLR(optimizer, **argv)
    elif scheduler_type == 'exponential':
        return lr_scheduler.ExponentialLR(optimizer, **argv)
    elif scheduler_type == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, **argv)
    elif scheduler_type == 'cosine-warmup':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **argv)
    elif scheduler_type == 'poly':
        return lr_scheduler.PolynomialLR(optimizer, **argv)
    raise NotImplementedError("Unrecognized scheduler type:{}".format(scheduler_type))

def get_optimizer(parameters,
        optimizer_type:Literal['adamw','adam','prodigy'],
        **argv):
    optimizer_type = optimizer_type.lower()
    if optimizer_type == 'adamw':
        return AdamW(parameters, **argv)
    elif optimizer_type == 'adam':
        return Adam(parameters, **argv)
    elif optimizer_type == 'prodigy':
        return Prodigy(parameters, **argv)
    else:
        raise NotImplementedError("Unrecognized Optimizer type:{}".format(optimizer_type))
