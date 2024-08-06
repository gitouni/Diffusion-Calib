import os
import shutil
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader
from dataset import BaseKITTIDataset, PertubKITTIDataset, KITTIBatchSampler
from models.denoiser import Surrogate
from models.diffusion_scheduler import DiffusionScheduler
from models.lr_scheduler import get_lr_scheduler
from models.loss import get_pcd_loss, geodesic_loss
from tqdm import tqdm
import yaml
from models.util import se3
from models.util.transform import inv_pose
from core.logger import LogTracker
from core.tools import load_checkpoint, save_checkpoint
import logging
from pathlib import Path
from typing import Dict, Callable


def get_dataloader(train_base_dataset_argv:Dict, train_dataset_argv:Dict,
        val_base_dataset_argv:Dict, val_dataset_argv:Dict,
        train_dataloader_argv:Dict, val_dataloader_argv:Dict):
    train_base_dataset = BaseKITTIDataset(**train_base_dataset_argv)
    val_base_dataset = BaseKITTIDataset(**val_base_dataset_argv)
    train_dataset = PertubKITTIDataset(train_base_dataset, **train_dataset_argv)
    val_dataset = PertubKITTIDataset(val_base_dataset, **val_dataset_argv)
    train_dataloader_argv['batch_sampler'] = KITTIBatchSampler(len(train_base_dataset.kitti_datalist), train_base_dataset.sep, **train_dataloader_argv['batch_sampler'])
    val_dataloader_argv['batch_sampler'] = KITTIBatchSampler(len(val_base_dataset.kitti_datalist), val_base_dataset.sep, **val_dataloader_argv['batch_sampler'])
    if hasattr(train_dataset, 'collate_fn'):
        train_dataloader_argv['collate_fn'] = getattr(train_dataset, 'collate_fn')
    if hasattr(val_dataset, 'collate_fn'):
        val_dataloader_argv['collate_fn'] = getattr(val_dataset, 'collate_fn')
    train_dataloader = DataLoader(train_dataset, **train_dataloader_argv)
    val_dataloader = DataLoader(val_dataset, **val_dataloader_argv)
    return train_dataloader, val_dataloader

@torch.inference_mode()
def val_epoch(val_loader:DataLoader, surrogate_model:nn.Module, diffusion_scheduler:DiffusionScheduler,
              pcd_loss_func:Callable, loss_weigths:Dict[str,float], logger:logging.Logger, opt:Dict):
    surrogate_model.eval()
    total_loss = 0
    logger.info("Validation:")
    iterator = tqdm(val_loader, desc='val')
    tracker = LogTracker('R','T','pcd','loss')
    with iterator:
        N_valid = len(val_loader)
        for i, batch in enumerate(val_loader):
            img = batch['img'].to(opt['device'])
            pcd = batch['uncalib_pcd'].to(opt['device'])
            gt_se3 = batch['gt'].to(opt['device'])  # transform uncalibrated_pcd to calibrated_pcd
            gt_pcd = se3.transform(gt_se3, pcd)
            camera_info = batch['camera_info']
            B = gt_se3.shape[0]
            H_t = torch.eye(4).unsqueeze(0).expand(B, -1, -1).to(opt['device'])  # [B, 4, 4] from I to H_0
            for t in range(diffusion_scheduler.num_steps, 0, -1):  # [T, T-1, ..., 1]
                X_t = (H_t[:, :3, :3] @ pcd + H_t[:, :3, [3]]) # [B, 3, N]
                pred_x = surrogate_model.forward(img, X_t, camera_info)
                delta_H_t = se3.exp(pred_x)  # (B, 4, 4)
                H_0 = delta_H_t @ H_t  # accumulate transformations

                gamma0 = diffusion_scheduler.gamma0[t]
                gamma1 = diffusion_scheduler.gamma1[t]
                H_t = se3.exp(gamma0 * se3.log(H_0) + gamma1 * se3.log(H_t))

                ### noise
                if opt['add_noise'] and t > 1:
                    alpha_bar = diffusion_scheduler.alpha_bars[t]
                    alpha_bar_ = diffusion_scheduler.alpha_bars[t-1]
                    beta = diffusion_scheduler.betas[t]
                    cc = ((1 - alpha_bar_) / (1.- alpha_bar)) * beta
                    scale = torch.cat([torch.ones(3) * opt['sigma_r'], torch.ones(3) * opt['sigma_t']])[None].to(opt['device'])  # [1, 6]
                    noise = torch.sqrt(cc) * scale * torch.randn(B, 6).to(opt['device'])  # [B, 6]
                    H_noise = se3.exp(noise)
                    H_t = H_noise @ H_t  # [B, 4, 4]
            X_t = (H_t[:, :3, :3] @ pcd + H_t[:, :3, [3]])  # final transform
            R_loss, t_loss = geodesic_loss(H_t, gt_se3)
            pcd_loss = pcd_loss_func(X_t, gt_pcd)
            loss = loss_weigths['R'] * R_loss + loss_weigths['t'] * t_loss + loss_weigths['pcd'] * pcd_loss
            if torch.isnan(loss).sum() > 0:
                logger.warn("nan value detected, skip this batch.")
                N_valid -= 1
            total_loss += loss
            tracker.update('R',R_loss.item())
            tracker.update('T',t_loss.item())
            tracker.update('pcd',pcd_loss.item())
            tracker.update('loss',loss.item())
            iterator.set_postfix(tracker.result())
            iterator.update(1)
            if (i+1) % opt['log_per_iter'] == 0 or (i+1) == len(val_loader):
                logger.info("\tBatch:{}|{}: {}".format(i+1, len(val_loader), tracker.result()))
    assert N_valid > 0, "Fatal Error, no valid batch!"
    return total_loss / N_valid



def main(config:Dict, config_path:str):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = config['device']
    # torch.backends.cudnn.benchmark=True
    # torch.backends.cudnn.enabled = False
    surrogate_model = Surrogate(**config['model']['surrogate']).to(device)
    train_diff_scheduler = DiffusionScheduler(config['model']['diffusion_scheduler']['train'])
    val_diff_scheduler = DiffusionScheduler(config['model']['diffusion_scheduler']['val'])
    scheduler_argv = config['model']['diffusion_scheduler']
    sigma_r, sigma_t = scheduler_argv['train']['sigma_r'], scheduler_argv['train']['sigma_t']
    dataset_argv = config['dataset']['train']
    train_dataloader, val_dataloader = get_dataloader(dataset_argv['dataset']['train']['base'], dataset_argv['dataset']['train']['main'],
        dataset_argv['dataset']['val']['base'], dataset_argv['dataset']['val']['main'],
        dataset_argv['dataloader']['args'], dataset_argv['dataloader']['val_args'])
    optimizer = torch.optim.Adam(surrogate_model.parameters(), **config['optimizer']['args'])
    clip_grad = config['optimizer']['max_grad']
    scheduler = get_lr_scheduler(optimizer, config['scheduler']['type'], **config['scheduler']['args'])
    pcd_loss_func = get_pcd_loss(config['loss']['pcd']['type'], **config['loss']['pcd']['args'])
    loss_weigths = config['loss']['weights']
    run_argv = config['run']
    path_argv = config['path']
    experiment_dir = Path(path_argv['base_dir'])
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(path_argv['name'])
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath(path_argv['checkpoint'])
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath(path_argv['log'])
    log_dir.mkdir(exist_ok=True)
    shutil.copyfile(config_path, str(log_dir.joinpath(os.path.basename(config_path))))  # copy the config file
    # logger
    logger = logging.getLogger(path_argv['log'])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_mode = 'a' if path_argv['resume'] is not None else 'w'
    file_handler = logging.FileHandler(str(log_dir) + '/train.log', mode=logger_mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('start traing')
    logger.info('args:')
    logger.info(args)
    if path_argv['resume'] is not None:
        start_epoch, best_loss = load_checkpoint(path_argv['resume'], surrogate_model, optimizer, scheduler)
        logger.info("Loaded checkpoint from {}, Start from Epoch {}".format(path_argv['resume'], start_epoch))
    else:
        start_epoch = 0
        best_loss = float('inf')
        logger.info("Start from scratch")
    ## training
    for epoch_idx in range(start_epoch, run_argv['n_epoch']):
        surrogate_model.train()
        iterator = tqdm(train_dataloader, desc='train')
        tracker = LogTracker('R','T','pcd','R_t','T_t','pcd_t','loss')
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

                taus = train_diff_scheduler.uniform_sample_t(B)
                alpha_bars = train_diff_scheduler.alpha_bars[taus].to(device).unsqueeze(1)  # [B, 1]
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

                expected_se3 = gt_se3 @ inv_pose(H_t_noise)
                R_loss, t_loss = geodesic_loss(pred_se3, expected_se3)
                pcd_loss = pcd_loss_func(se3.transform(pred_se3, X_t), gt_pcd)
                loss += loss_weigths['R'] * R_loss + loss_weigths['t'] * t_loss + loss_weigths['pcd'] * pcd_loss
                tracker.update('R_t',R_loss.item())
                tracker.update('T_t',t_loss.item())
                tracker.update('pcd_t',pcd_loss.item())

                pred_x = surrogate_model(img, pcd, camera_info)
                pred_se3 = se3.exp(pred_x)

                R_loss, t_loss = geodesic_loss(pred_se3, gt_se3)
                pcd_loss = pcd_loss_func(se3.transform(pred_se3, pcd), gt_pcd)
                loss += loss_weigths['R'] * R_loss + loss_weigths['t'] * t_loss + loss_weigths['pcd'] * pcd_loss
                if torch.isnan(loss).sum() > 0:
                    logger.warn("nan detected in loss, skip this batch.")
                    continue
                tracker.update('R',R_loss.item())
                tracker.update('T',t_loss.item())
                tracker.update('pcd',pcd_loss.item())
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(surrogate_model.parameters(), clip_grad)
                optimizer.step()
                tracker.update('loss',loss.item())
                iterator.set_postfix(tracker.result())
                iterator.update(1)
                if (i+1) % run_argv['log_per_iter'] == 0:
                    logger.info("\tBatch {}|{}: {}".format(i+1, len(train_dataloader), tracker.result()))
            logger.info("Epoch {}|{}: {}".format(epoch_idx+1, run_argv['n_epoch'], tracker.result()))
            scheduler.step()
            opt = {"device": device, 'sigma_r':scheduler_argv['val']['sigma_r'], 'sigma_t':scheduler_argv['val']['sigma_t'],
                'add_noise': scheduler_argv['val']['add_noise'],'log_per_iter':run_argv['log_per_iter']}
            if (epoch_idx + 1) % run_argv['val_per_epoch'] == 0:
                val_loss = val_epoch(val_dataloader, surrogate_model, val_diff_scheduler, pcd_loss_func, loss_weigths, logger, opt)
                if val_loss < best_loss:
                    logger.info("Find Best Model at Epoch {} prev | curr best loss: {} | {}".format(epoch_idx+1, best_loss, val_loss))
                    best_loss = val_loss
                    save_checkpoint(str(checkpoints_dir.joinpath('best_model.pth')), epoch_idx, best_loss, surrogate_model, optimizer, scheduler)
            save_checkpoint(str(checkpoints_dir.joinpath('last_model.pth')), epoch_idx, best_loss, surrogate_model, optimizer, scheduler)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="cfg/kitti_surrogate.yml", type=str)
    args = parser.parse_args()
    config = yaml.load(open(args.config,'r'), yaml.SafeLoader)
    main(config, args.config)