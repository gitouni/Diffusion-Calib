import os
import shutil
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader
from dataset import PerturbDataset, SeqBatchSampler, __classdict__ as DatasetDict, DATASET_TYPE
from models.denoiser import __classdict__ as DenoiserDict, SURROGATE_TYPE
from models.diffuser import SE3Diffuser
from models.lr_scheduler import get_lr_scheduler, get_optimizer
from models.loss import geodesic_loss, get_loss
from tqdm import tqdm
import yaml
from models.util import se3
from core.logger import LogTracker, fmt_time, print_warning
from core.tools import load_checkpoint, save_checkpoint
import logging
from pathlib import Path
from typing import Dict


def get_dataloader(dataset_type:str, train_base_dataset_argv:Dict, train_dataset_argv:Dict,
        val_base_dataset_argv:Dict, val_dataset_argv:Dict,
        train_dataloader_argv:Dict, val_dataloader_argv:Dict):
    dataset_class = DatasetDict[dataset_type]
    train_base_dataset:DATASET_TYPE = dataset_class(**train_base_dataset_argv)
    val_base_dataset:DATASET_TYPE = dataset_class(**val_base_dataset_argv)
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
def val_epoch(val_loader:DataLoader, diffuser:SE3Diffuser, logger:logging.Logger, device, log_per_iter:int):
    diffuser.model.eval()
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
            camera_info = batch['camera_info']
            x0_se3 = diffuser.sampling((img, pcd, init_extran, camera_info), return_intermediate=False)
            R_loss, t_loss = geodesic_loss(x0_se3, gt_se3)
            loss = R_loss + t_loss
            if torch.isnan(loss).sum() > 0:
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
    checkpoints_dir = experiment_dir.joinpath(path_argv['checkpoint'])
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath(path_argv['log'])
    log_dir.mkdir(exist_ok=True)
    save_yaml_path = str(log_dir.joinpath(os.path.basename(config_path)))
    print_warning("config file saved to {}.".format(save_yaml_path))
    yaml.safe_dump(config, open(save_yaml_path,'w'))
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = config['device']
    # torch.backends.cudnn.benchmark=True
    # torch.backends.cudnn.enabled = False
    surrogate_model:SURROGATE_TYPE = DenoiserDict[config['surrogate']['type']](**config['surrogate']['argv']).to(device)
    scheduler_argv = config['diffuser']
    diffuser = SE3Diffuser(surrogate_model, scheduler_argv['train'], scheduler_argv['val'])
    loss_func = get_loss(config['loss']['type'], **config['loss']['args'])
    diffuser.set_loss(loss_func)
    dataset_argv = config['dataset']['train']
    dataset_type = config['dataset']['type']
    train_dataloader, val_dataloader = get_dataloader(dataset_type, dataset_argv['dataset']['train']['base'], dataset_argv['dataset']['train']['main'],
        dataset_argv['dataset']['val']['base'], dataset_argv['dataset']['val']['main'],
        dataset_argv['dataloader']['args'], dataset_argv['dataloader']['val_args'])
    optimizer = get_optimizer(filter(lambda p: p.requires_grad, surrogate_model.parameters()), config['optimizer']['type'], **config['optimizer']['args'])
    clip_grad = config['optimizer']['max_grad']
    scheduler = get_lr_scheduler(optimizer, config['scheduler']['type'], **config['scheduler']['args'])
    # logger
    logger = logging.getLogger(path_argv['log'])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_mode = 'a' if path_argv['resume'] is not None else 'w'
    file_handler = logging.FileHandler(str(log_dir) + '/train_{}.log'.format(fmt_time()), mode=logger_mode)
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
        tracker = LogTracker('R','t','loss')
        with iterator:
            for i, batch in enumerate(train_dataloader):
                # model prediction
                img = batch['img'].to(device)
                pcd = batch['pcd'].to(device)
                init_extran = batch['extran'].to(device)
                gt_se3 = batch['gt'].to(device)  # transform uncalibrated_pcd to calibrated_pcd
                camera_info = batch['camera_info']
                loss, x0_hat = diffuser(gt_se3, [img, pcd, init_extran, camera_info])
                with torch.inference_mode():
                    R_loss, t_loss = geodesic_loss(se3.exp(x0_hat), gt_se3)
                if torch.isnan(loss).sum() > 0:
                    logger.warning("nan detected in loss, skip this batch.")
                    continue
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(surrogate_model.parameters(), clip_grad)
                optimizer.step()
                tracker.update('R', R_loss.item())
                tracker.update('t', t_loss.item())
                tracker.update('loss',loss.item())
                iterator.set_postfix(tracker.result())
                iterator.update(1)
                if (i+1) % run_argv['log_per_iter'] == 0:
                    logger.info("\tBatch {}|{}: {}".format(i+1, len(train_dataloader), tracker.result()))
            logger.info("Epoch {}|{}: {}".format(epoch_idx+1, run_argv['n_epoch'], tracker.result()))
            scheduler.step()
            if (epoch_idx + 1) % run_argv['val_per_epoch'] == 0:
                val_loss = val_epoch(val_dataloader, diffuser, logger, device, run_argv['log_per_iter'])
                if val_loss < best_loss:
                    logger.info("Find Best Model at Epoch {} prev | curr best loss: {} | {}".format(epoch_idx+1, best_loss, val_loss))
                    best_loss = val_loss
                    save_checkpoint(str(checkpoints_dir.joinpath('best_model.pth')), epoch_idx, best_loss, surrogate_model, optimizer, scheduler)
            save_checkpoint(str(checkpoints_dir.joinpath('last_model.pth')), epoch_idx, best_loss, surrogate_model, optimizer, scheduler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', default="cfg/dataset/kitti_large.yml", type=str)
    parser.add_argument("--model_config",type=str,default="cfg/model/calibnet.yml")
    parser.add_argument("--common_config",type=str,default="cfg/common.yml")
    parser.add_argument("--mode_config",type=str,default="cfg/mode/nlsd.yml")
    args = parser.parse_args()
    dataset_config:Dict = yaml.safe_load(open(args.dataset_config,'r'))
    model_config:Dict = yaml.safe_load(open(args.model_config,'r'))
    mode_config:Dict = yaml.safe_load(open(args.mode_config,'r'))
    dataset_name = dataset_config['name']
    model_name = model_config['name']
    mode_name = mode_config['name']
    dataset_config.pop('name')
    model_config.pop('name')
    mode_config.pop('name')
    config:Dict = yaml.safe_load(open(args.common_config,'r'))
    config['path']['base_dir'] = config['path']['base_dir'].format(dataset=dataset_name, model=model_name, mode=mode_name)
    config['path']['pretrain'] = config['path']['pretrain'].format(dataset=dataset_name, model=model_name, mode=mode_name)
    config.update(dataset_config)
    config.update(model_config)
    config.update(mode_config)
    main(config, '{}_{}_{}.yml'.format(dataset_name, mode_name, model_name))