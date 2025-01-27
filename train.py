import os
from pathlib import Path
import shutil
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader
from dataset import __classdict__ as DatasetDict
from dataset import PerturbDataset, SeqBatchSampler
from models.denoiser import Surrogate, Denoiser, RGGDenoiser, RAFTDenoiser, __classdict__ as DenoiserDict
from models.diffuser import Diffuser
from models.lr_scheduler import get_lr_scheduler, get_optimizer
from models.loss import get_loss, geodesic_loss
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

def get_dataloader(dataset_type:str, train_base_dataset_argv:Dict, train_dataset_argv:Dict,
        val_base_dataset_argv:Dict, val_dataset_argv:Dict,
        train_dataloader_argv:Dict, val_dataloader_argv:Dict):
    dataset_class = DatasetDict[dataset_type]
    train_base_dataset = dataset_class(**train_base_dataset_argv)
    val_base_dataset = dataset_class(**val_base_dataset_argv)
    train_dataset = PerturbDataset(train_base_dataset, **train_dataset_argv)
    val_dataset = PerturbDataset(val_base_dataset, **val_dataset_argv)
    train_dataloader_argv['batch_sampler'] = SeqBatchSampler(*train_base_dataset.get_seq_params(), **train_dataloader_argv['batch_sampler'])
    val_dataloader_argv['batch_sampler'] = SeqBatchSampler(*val_base_dataset.get_seq_params(),  **val_dataloader_argv['batch_sampler'])
    if hasattr(train_dataset, 'collate_fn'):
        train_dataloader_argv['collate_fn'] = getattr(train_dataset, 'collate_fn')
    if hasattr(val_dataset, 'collate_fn'):
        val_dataloader_argv['collate_fn'] = getattr(val_dataset, 'collate_fn')
    train_dataloader = DataLoader(train_dataset, **train_dataloader_argv)
    val_dataloader = DataLoader(val_dataset, **val_dataloader_argv)
    return train_dataloader, val_dataloader

@torch.inference_mode()
def val_epoch(val_loader:DataLoader, diffuser:Diffuser, logger:logging.Logger, device, log_per_iter:int):
    diffuser.x0_fn.model.eval()
    total_loss = 0
    logger.info("Validation:")
    iterator = tqdm(val_loader, desc='val')
    tracker = LogTracker('R','T','loss')
    with iterator:
        N_valid = len(val_loader)
        for i, batch in enumerate(val_loader):
            img = batch['img'].to(device)
            pcd = batch['pcd'].to(device)
            init_extran = batch['extran'].to(device)
            gt_se3 = batch['gt'].to(device)  # transform uncalibrated_pcd to calibrated_pcd
            gt_x = se3.log(gt_se3)
            camera_info = batch['camera_info']
            x0_hat = diffuser.sample_fn(torch.zeros_like(gt_x), (img, pcd, init_extran, camera_info))
            x0_se3 = se3.exp(x0_hat)
            R_loss, t_loss = geodesic_loss(x0_se3, gt_se3)
            if diffuser.seq_loss:
                loss = diffuser.loss_fn([x0_hat], gt_x)
            else:
                loss = diffuser.loss_fn(x0_hat, gt_x)
            if torch.isnan(R_loss).sum() + torch.isnan(t_loss).sum() + torch.isnan(loss).sum() > 0:
                logger.warning("nan value detected, validation failed.")
                N_valid -= 1
                iterator.set_postfix(state='nan')
                iterator.update(1)
                continue
            total_loss += loss
            tracker.update('R',R_loss.item())
            tracker.update('T',t_loss.item())
            tracker.update('loss',loss.item())
            iterator.set_postfix(tracker.result())
            iterator.update(1)
            if (i+1) % log_per_iter == 0 or (i+1) == len(val_loader):
                logger.info("\tBatch:{}|{}: {}".format(i+1, len(val_loader), tracker.result()))
    assert N_valid > 0, "Fatal Error, no valid batch!"
    return total_loss / N_valid



def main(config:Dict, config_path:str):
    run_argv = config['run']
    path_argv = config['path']
    experiment_dir = Path(path_argv['base_dir'])
    experiment_dir.mkdir(exist_ok=True, parents=True)
    experiment_dir = experiment_dir.joinpath(path_argv['name'])
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath(path_argv['checkpoint'])
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath(path_argv['log'])
    log_dir.mkdir(exist_ok=True)
    yaml.safe_dump(config, open(str(log_dir.joinpath(os.path.basename(config_path))),'w'))
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = config['device']
    # torch.backends.cudnn.benchmark=True
    # torch.backends.cudnn.enabled = False
    surrogate_model:Surrogate = DenoiserDict[config['model']['surrogate']['type']](**config['model']['surrogate']['argv']).to(device)
    if config['model']['surrogate']['type'] == 'LCCRAFT':
        denoiser_class = RAFTDenoiser
    elif config['model']['surrogate']['type'] == 'RGGNet':
        denoiser_class = RGGDenoiser
    else:
        denoiser_class = Denoiser
    denoiser = denoiser_class(surrogate_model)
    diffuser = Diffuser(denoiser, **config['model']['diffuser'])
    dataset_argv = config['dataset']['train']
    dataset_type = config['dataset']['type']
    train_dataloader, val_dataloader = get_dataloader(dataset_type, dataset_argv['dataset']['train']['base'], dataset_argv['dataset']['train']['main'],
        dataset_argv['dataset']['val']['base'], dataset_argv['dataset']['val']['main'],
        dataset_argv['dataloader']['args'], dataset_argv['dataloader']['val_args'])
    optimizer = get_optimizer(filter(lambda p: p.requires_grad, surrogate_model.parameters()), config['optimizer']['type'], **config['optimizer']['args'])
    clip_grad = config['optimizer']['max_grad']
    scheduler = get_lr_scheduler(optimizer, config['scheduler']['type'], **config['scheduler']['args'])
    loss_func = get_loss(config['loss']['type'], **config['loss']['args'])
    diffuser.set_loss(loss_func)
    diffuser.set_new_noise_schedule(device)
    # logger
    steps = config['model']['diffuser']['sampling_argv']['steps']
    name = "{}_{}".format(diffuser.sampling_type, steps)
    logger = logging.getLogger(path_argv['log'])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_mode = 'a' if path_argv['resume'] is not None else 'w'
    file_handler = logging.FileHandler(str(log_dir) + '/train_{}_{}.log'.format(name,fmt_time()), mode=logger_mode)
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
        start_epoch = 1
        best_loss = float('inf')
        logger.info("Start from scratch")
    ## training
    for epoch_idx in range(start_epoch, run_argv['n_epoch']+1):
        diffuser.x0_fn.model.train()
        iterator = tqdm(train_dataloader, desc='train')
        tracker = LogTracker('R','T','loss')
        with iterator:
            for i, batch in enumerate(train_dataloader):
                # model prediction
                img = batch['img'].to(device)
                pcd = batch['pcd'].to(device)
                init_extran = batch['extran'].to(device)
                gt_se3 = batch['gt'].to(device)  # transform uncalibrated_pcd to calibrated_pcd
                camera_info = batch['camera_info']
                gt_delta_x = se3.log(gt_se3)  # (B, 6)
                optimizer.zero_grad()
                loss, x0_hat = diffuser.forward(gt_delta_x, (img, pcd, init_extran, camera_info))
                # R_loss, t_loss = geodesic_loss(se3.exp(x0_hat), gt_se3)
                # loss = R_loss + t_loss
                if torch.isnan(loss).sum() > 0:
                    logger.warning("nan detected, skip this step.")
                    iterator.set_postfix(state='nan')
                    iterator.update(1)
                    optimizer.zero_grad()
                    continue
                if isinstance(diffuser.x0_fn, RGGDenoiser):
                    with torch.enable_grad():
                        ELBO = diffuser.x0_fn.loss(x0_hat, (img, pcd, init_extran, camera_info))
                    loss = loss + ELBO
                loss.backward()
                try:
                    nn.utils.clip_grad_norm_(diffuser.x0_fn.model.parameters(), clip_grad, norm_type=2, error_if_nonfinite=True)  # avoid gradient explosion
                    optimizer.step()
                except:
                    logger.warning("nan detected in grad, skip batch.")
                    iterator.update(1)
                    optimizer.zero_grad()
                    continue
                with torch.inference_mode():
                    R_loss, t_loss = geodesic_loss(se3.exp(x0_hat), gt_se3)
                tracker.update('R',R_loss.item())
                tracker.update('T',t_loss.item())
                tracker.update('loss',loss.item())
                iterator.set_postfix(tracker.result())
                iterator.update(1)
                if (i+1) % run_argv['log_per_iter'] == 0:
                    logger.info("\tBatch {}|{}: {}".format(i+1, len(train_dataloader), tracker.result()))
            logger.info("Epoch {}|{}: {}".format(epoch_idx, run_argv['n_epoch'], tracker.result()))
            scheduler.step()
            save_checkpoint(str(checkpoints_dir.joinpath('last_model.pth')), epoch_idx, best_loss, diffuser.x0_fn.model, optimizer, scheduler)
        if epoch_idx % run_argv['val_per_epoch'] == 0:
            val_loss = val_epoch(val_dataloader, diffuser, logger, device, run_argv['log_per_iter'])
            if val_loss < best_loss:
                logger.info("Find Best Model at Epoch {} prev | curr best loss: {} | {}".format(epoch_idx, best_loss, val_loss))
                best_loss = val_loss
                save_checkpoint(str(checkpoints_dir.joinpath('best_model.pth')), epoch_idx, best_loss, diffuser.x0_fn.model, optimizer, scheduler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', default="cfg/dataset/kitti_large.yml", type=str)
    parser.add_argument("--model_config",type=str,default="cfg/unipc_model/main.yml")
    args = parser.parse_args()
    dataset_config = yaml.load(open(args.dataset_config,'r'), yaml.SafeLoader)
    config = yaml.load(open(args.model_config,'r'), yaml.SafeLoader)
    config.update(dataset_config)
    main(config, args.model_config)