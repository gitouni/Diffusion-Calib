from .utils import registry
import os
import torch
from typing import Dict

MODELS = registry.Registry('models')


def build_model_from_cfg(cfg, **kwargs):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT): 
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return MODELS.build(cfg, **kwargs)



def load_model(base_model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError(
            'no checkpoint file from path %s...' % ckpt_path)

    # load state dict
    state_dict:Dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k,
                     v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k,
                     v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt, strict=True)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, Dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    return epoch, metrics