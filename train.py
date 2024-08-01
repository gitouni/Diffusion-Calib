import os
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import BaseKITTIDataset, PertubKITTIDataset, subset_split
from models.denoiser import Surrogate
from models.diffusion_scheduler import DiffusionScheduler
from models.lr_scheduler import get_lr_scheduler
from models.loss import get_pcd_loss, geodesic_loss
from tqdm import tqdm
from collections import defaultdict
import yaml
from util import se3
from util.transform import inv_pose
from typing import Dict, Optional


def get_dataloader(base_dataset_argv:Dict, dataset_argv:Dict, train_dataloader_argv:Dict, val_dataloader_argv:Dict,
        validation_split:float, seed:Optional[int]=None):
    base_dataset = BaseKITTIDataset(**base_dataset_argv)
    phase_dataset = PertubKITTIDataset(base_dataset, **dataset_argv)
    train_length = int((1 - validation_split) * len(phase_dataset))
    val_length = len(phase_dataset) - train_length
    train_dataset, val_dataset = subset_split(phase_dataset, [train_length, val_length], seed)
    if hasattr(phase_dataset, 'collate_fn'):
        train_dataloader_argv['collate_fn'] = getattr(phase_dataset, 'collate_fn')
        val_dataloader_argv['collate_fn'] = getattr(phase_dataset, 'collate_fn')
    train_dataloader = DataLoader(train_dataset, **train_dataloader_argv)
    val_dataloader = DataLoader(val_dataset, **val_dataloader_argv)
    return train_dataloader, val_dataloader


def main(config:Dict):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = config['device']
    # torch.backends.cudnn.benchmark=True
    # torch.backends.cudnn.enabled = False
    surrogate_model = Surrogate(**config['model']['surrogate']).to(device)
    diffusion_scheduler = DiffusionScheduler(config['model']['diffusion_scheduler']['args'])
    sigma_r, sigma_t =  config['model']['diffusion_scheduler']['sigma_r'], config['model']['diffusion_scheduler']['sigma_t']
    dataset_argv = config['dataset']['train']
    train_dataloader, val_dataloader = get_dataloader(dataset_argv['dataset']['base'], dataset_argv['dataset']['main'],
        dataset_argv['dataloader']['args'], dataset_argv['dataloader']['val_args'], dataset_argv['validation_split'], config['seed'])
    optimizer = torch.optim.Adam(surrogate_model.parameters(), **config['optimizer']['args'])
    scheduler = get_lr_scheduler(optimizer, config['scheduler']['type'], **config['scheduler']['args'])
    pcd_loss_func = get_pcd_loss(config['loss']['pcd']['type'], **config['loss']['pcd']['args'])
    loss_weigths = config['loss']['weights']
    run_argv = config['run']
    ## training
    for epoch_idx in range(run_argv['n_epoch']):
        surrogate_model.train()
        iterator = tqdm(train_dataloader, desc='train')
        with iterator:
            for i, batch in enumerate(train_dataloader):
                # model prediction
                img = batch['img'].to(device)
                pcd = batch['uncalib_pcd'].to(device)
                gt_se3 = batch['gt'].to(device)  # transform uncalibrated_pcd to calibrated_pcd
                gt_pcd = se3.transform(gt_se3, pcd)
                camera_info = batch['camera_info']
                B = gt_se3.shape[0]

                ### SE(3) diffusion process
                H_T = torch.eye(4).unsqueeze(0).expand(B, -1, -1).to(device)
                H_0 = gt_se3

                taus = diffusion_scheduler.uniform_sample_t(B)
                alpha_bars = diffusion_scheduler.alpha_bars[taus].to(device).unsqueeze(1)  # [B, 1]
                H_t = se3.exp((1. - torch.sqrt(alpha_bars)) * se3.log(H_T @ inv_pose(H_0))) @ H_0

                ### add noise
                scale = torch.cat([torch.ones(3) * sigma_r, torch.ones(3) * sigma_t]).unsqueeze(0).to(device)  # [1, 6]
                noise = torch.sqrt(1. - alpha_bars) * scale * torch.clamp(torch.randn(B, 6), -3, 3).to(device)  # [B, 6]
                H_noise = se3.exp(noise)
                H_t_noise = H_noise @ H_t  # [B, 4, 4]

                T_t_R = H_t_noise[:, :3, :3]  # [B, 3, 3]
                T_t_t = H_t_noise[:, :3, [3]]  # [B, 3, 1]

                X_t = (T_t_R @ pcd + T_t_t)  # [B, 3, N]

                pred_x = surrogate_model(img, X_t, camera_info)
                pred_se3 = se3.exp(pred_x)
                loss = 0
                loss_dict = dict()
                expected_se3 = gt_se3 @ inv_pose(H_t_noise)
                R_loss, t_loss = geodesic_loss(pred_se3, expected_se3)
                pcd_loss = pcd_loss_func(se3.transform(pred_se3, X_t), gt_pcd)
                loss += loss_weigths['R'] * R_loss + loss_weigths['t'] * t_loss + loss_weigths['pcd'] * pcd_loss
                loss_dict.update({"R_t":R_loss.item(), "T_t":t_loss.item(), 'pcd_t':pcd_loss.item()})

                pred_x = surrogate_model(img, pcd, camera_info)
                pred_se3 = se3.exp(pred_x)

                R_loss, t_loss = geodesic_loss(pred_se3, gt_se3)
                pcd_loss = pcd_loss_func(se3.transform(pred_se3, pcd), gt_pcd)
                loss += loss_weigths['R'] * R_loss + loss_weigths['t'] * t_loss + loss_weigths['pcd'] * pcd_loss
                loss_dict.update({"R":R_loss.item(), "T":t_loss.item(), 'pcd':pcd_loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iterator.set_postfix(loss_dict)
                iterator.update(1)
                # print("=== Train. Epoch [%d], losses: %1.3f ===" % (epoch_idx, loss.item()))

        scheduler.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="cfg/kitti.yml", type=str)
    args = parser.parse_args()
    config = yaml.load(open(args.config,'r'), yaml.SafeLoader)
    main(config)