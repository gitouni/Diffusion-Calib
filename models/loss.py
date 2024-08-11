import torch.nn as nn
import torch
from typing import Literal, Tuple
from functools import partial
from .util import se3
from .util.transform import inv_pose
from .util.rotation_conversions import matrix_to_euler_angles

def se3_err(pred_se3:torch.Tensor, gt_se3:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_pred = inv_pose(pred_se3)
    delta_se3 = inv_pred @ gt_se3
    delta_euler = matrix_to_euler_angles(delta_se3[...,:3,:3], 'XYZ')
    delta_tsl = torch.abs(delta_se3[...,:3,3])
    return delta_euler, delta_tsl  # (B, 3), (B, 3)

def get_loss(loss_type:Literal['mae','smooth_mae','mse'], **argv):
    if loss_type == 'mae':
        return partial(mae, **argv)
    if loss_type == 'smooth_mae':
        return partial(mae, **argv)
    if loss_type == 'mse':
        return partial(mae, **argv)
    return NotImplementedError("Unrecognized loss_type:{}".format(loss_type))


def get_pcd_loss(loss_type:Literal['mae','smooth_mae','mse'], **argv):
    if loss_type == 'mae':
        return partial(mae, **argv)
    if loss_type == 'smooth_mae':
        return partial(mae, **argv)
    if loss_type == 'mse':
        return partial(mae, **argv)
    return NotImplementedError("Unrecognized loss_type:{}".format(loss_type))

def geodesic_loss(pred_se3:torch.Tensor, gt_se3:torch.Tensor):
    assert pred_se3.ndim == gt_se3.ndim == 3, "pred_se3 and gt_se3 must be [B, 4, 4]"
    pred_R, gt_R = pred_se3[:,:3,:3], gt_se3[:, :3,:3]  # (B, 3, 3)
    pred_t, gt_t = pred_se3[:,:3,3], gt_se3[:,:3, 3]  # (B, 3)
    trace = torch.einsum('aij,aij->a', pred_R, gt_R)  # trace of pred_R @ gt_R.T
    angular_err = torch.arccos(torch.clip((trace - 1.0) / 2.0, -1.0, 1.0)).mean()
    translation_err = torch.abs(pred_t - gt_t).mean()
    return angular_err, translation_err

def mae(pred, target, reduction='mean'):
    loss = nn.L1Loss(reduction=reduction)
    return loss(pred, target)

def smooth_mae(pred, target, beta:float, reduction='mean'):
    loss = nn.SmoothL1Loss(reduction=reduction, beta=beta)
    return loss(pred, target)

def mse(pred, target, reduction='mean'):
    loss = nn.MSELoss(reduction=reduction)
    return loss(pred, target)