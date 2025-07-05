"""
Train VAE of the RGGNet
"""
import os
import shutil
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader
from dataset import __classdict__ as DatasetDict
from dataset import SeqBatchSampler
from models.rggnet.vae import VanillaVAE as VAE
from models.tools.core import DepthImgGenerator
from models.lr_scheduler import get_lr_scheduler
from tqdm import tqdm
import yaml
from models.util import se3
from core.logger import LogTracker, fmt_time, print_warning
from core.tools import load_checkpoint, save_checkpoint
import logging
from pathlib import Path
from typing import Dict, Union, Iterable

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

def get_dataloader(dataset_type:str, train_base_dataset_argv:Dict,
        val_base_dataset_argv:Dict, 
        train_dataloader_argv:Dict, val_dataloader_argv:Dict):
    dataset_class = DatasetDict[dataset_type]
    train_base_dataset = dataset_class(**train_base_dataset_argv)
    val_base_dataset = dataset_class(**val_base_dataset_argv)
    train_dataloader_argv['batch_sampler'] = SeqBatchSampler(*train_base_dataset.get_seq_params(), **train_dataloader_argv['batch_sampler'])
    val_dataloader_argv['batch_sampler'] = SeqBatchSampler(*val_base_dataset.get_seq_params(),  **val_dataloader_argv['batch_sampler'])
    if hasattr(train_base_dataset, 'collate_fn'):
        train_dataloader_argv['collate_fn'] = getattr(train_base_dataset, 'collate_fn')
    if hasattr(val_base_dataset, 'collate_fn'):
        val_dataloader_argv['collate_fn'] = getattr(val_base_dataset, 'collate_fn')
    train_dataloader = DataLoader(train_base_dataset, **train_dataloader_argv)
    val_dataloader = DataLoader(val_base_dataset, **val_dataloader_argv)
    return train_dataloader, val_dataloader

@torch.inference_mode()
def val_epoch(val_loader:DataLoader, vae:VAE, depthgen_argv:Dict, vae_kld_weight:float, logger:logging.Logger, device, log_per_iter:int):
    vae.eval()
    total_loss = 0
    logger.info("Validation:")
    iterator = tqdm(val_loader, desc='val')
    tracker = LogTracker('recon','kld','loss')
    depth_generator = DepthImgGenerator(**depthgen_argv)
    with iterator:
        N_valid = len(val_loader)
        for i, batch in enumerate(val_loader):
            img = batch['img'].to(device)
            pcd = batch['pcd'].to(device)
            extran = batch['extran'].to(device)
            camera_info = batch['camera_info']
            pcd_tf = se3.transform(extran, pcd)
            depth = depth_generator.project(pcd_tf, camera_info)
            x_img = torch.cat([img, depth], dim=1)  # (B, 4, H, W)
            x_est, mu, log_var = vae.forward(x_img)
            recon_loss = vae.reconstruction_loss(x_est, depth)
            kld_loss = vae.kld_loss(mu, log_var)
            loss = recon_loss.item() + vae_kld_weight * kld_loss.item()
            total_loss += loss
            tracker.update('recon',recon_loss.item())
            tracker.update('kld',kld_loss.item())
            tracker.update('loss',loss)
            iterator.set_postfix(tracker.result())
            iterator.update(1)
            if (i+1) % log_per_iter == 0 or (i+1) == len(val_loader):
                logger.info("\tBatch:{}|{}: {}".format(i+1, len(val_loader), tracker.result()))
    assert N_valid > 0, "Fatal Error, no valid batch!"
    return total_loss / N_valid



def main(config:Dict, config_filename:str):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = config['device']
    run_argv = config['run']
    path_argv = config['path']
    experiment_dir = Path(path_argv['base_dir'])
    experiment_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir = experiment_dir.joinpath(path_argv['checkpoint'])
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath(path_argv['log'])
    log_dir.mkdir(exist_ok=True)
    save_yaml_path = str(log_dir.joinpath(config_filename))
    yaml.safe_dump(config, open(save_yaml_path,'w'))
    print_warning("config file saved to {}.".format(save_yaml_path))
    # torch.backends.cudnn.benchmark=True
    # torch.backends.cudnn.enabled = False
    vae = VAE(**config['vae']['argv']).to(device)
    vae_kld_weight:float = config['vae']['kld_weight']
    depthgen_argv:Dict = config['vae']['depthgen_argv']
    dataset_argv = config['dataset']['train']
    train_dataloader, val_dataloader = get_dataloader(config['dataset']['type'], dataset_argv['dataset']['train']['base'], dataset_argv['dataset']['val']['base'], 
        dataset_argv['dataloader']['args'], dataset_argv['dataloader']['val_args'])
    optimizer = torch.optim.Adam(vae.parameters(), **config['optimizer']['args'])
    clip_grad = config['optimizer']['max_grad']
    scheduler = get_lr_scheduler(optimizer, config['scheduler']['type'], **config['scheduler']['args'])
    # logger
    logger = logging.getLogger(path_argv['log'])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_mode = 'a' if path_argv['resume'] is not None else 'w'
    file_handler = logging.FileHandler(str(log_dir) + '/train_vae_{}.log'.format(fmt_time()), mode=logger_mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('start traing')
    logger.info('args:')
    logger.info(args)
    if path_argv['resume'] is not None:
        start_epoch, best_loss = load_checkpoint(path_argv['resume'], vae, optimizer, scheduler)
        logger.info("Loaded checkpoint from {}, Start from Epoch {}".format(path_argv['resume'], start_epoch))
    else:
        start_epoch = 1
        best_loss = float('inf')
        logger.info("Start from scratch")
    depth_generator = DepthImgGenerator(**depthgen_argv)
    ## training
    for epoch_idx in range(start_epoch, run_argv['n_epoch']+1):
        vae.train()
        iterator = tqdm(train_dataloader, desc='train')
        tracker = LogTracker('recon','kld','loss')
        with iterator:
            for i, batch in enumerate(train_dataloader):
                # model prediction
                img = batch['img'].to(device)
                pcd = batch['pcd'].to(device)
                extran = batch['extran'].to(device)
                camera_info = batch['camera_info']
                pcd_tf = se3.transform(extran, pcd)
                depth = depth_generator.project(pcd_tf, camera_info)
                x_img = torch.cat([img, depth], dim=1)  # (B, 4, H, W)
                x_est, mu, log_var = vae.forward(x_img)
                optimizer.zero_grad()
                recon_loss = vae.reconstruction_loss(x_est, depth)
                kld_loss = vae.kld_loss(mu, log_var)
                loss = recon_loss + vae_kld_weight * kld_loss
                # R_loss, t_loss = geodesic_loss(se3.exp(x0_hat), gt_se3)
                # loss = R_loss + t_loss
                if torch.isnan(loss).sum() > 0:
                    logger.warning("nan detected, training failed.")
                    iterator.set_postfix(state='nan')
                    iterator.update(1)
                    exit(1)
                loss.backward()
                nn.utils.clip_grad_norm_(vae.parameters(), max_norm=clip_grad, norm_type=2)  # avoid gradient explosion
                optimizer.step()
                tracker.update('recon',recon_loss.item())
                tracker.update('kld',kld_loss.item())
                tracker.update('loss',loss.item())
                iterator.set_postfix(tracker.result())
                iterator.update(1)
                if (i+1) % run_argv['log_per_iter'] == 0:
                    logger.info("\tBatch {}|{}: {}".format(i+1, len(train_dataloader), tracker.result()))
            logger.info("Epoch {}|{}: {}".format(epoch_idx, run_argv['n_epoch'], tracker.result()))
            scheduler.step()
            save_checkpoint(str(checkpoints_dir.joinpath('last_model.pth')), epoch_idx, best_loss, vae, optimizer, scheduler)
        if epoch_idx % run_argv['val_per_epoch'] == 0:
            val_loss = val_epoch(val_dataloader, vae, depthgen_argv, vae_kld_weight, logger, device, run_argv['log_per_iter'])
            if val_loss < best_loss:
                logger.info("Find Best Model at Epoch {} prev | curr best loss: {} | {}".format(epoch_idx, best_loss, val_loss))
                best_loss = val_loss
                save_checkpoint(str(checkpoints_dir.joinpath('best_model.pth')), epoch_idx, best_loss, vae, optimizer, scheduler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', default="cfg/dataset/nusc.yml", type=str)
    parser.add_argument("--model_config",type=str,default="cfg/model/vae.yml")
    parser.add_argument("--common_config",type=str,default="cfg/common.yml")
    args = parser.parse_args()
    dataset_config:Dict = yaml.safe_load(open(args.dataset_config,'r'))
    model_config:Dict = yaml.safe_load(open(args.model_config,'r'))
    dataset_name = dataset_config['name']
    model_name = model_config['name']
    dataset_config.pop('name')
    model_config.pop('name')
    config:Dict = yaml.safe_load(open(args.common_config,'r'))
    config['path']['base_dir'] = config['path']['base_dir'].format(dataset=dataset_name, model=model_name, mode='')
    config['path']['pretrain'] = config['path']['pretrain'].format(dataset=dataset_name, model=model_name, mode='')
    config.update(dataset_config)
    config.update(model_config)
    main(config, '{}_{}.yml'.format(dataset_name, model_name))