import os
import shutil
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader
from dataset import BaseKITTIDataset, KITTIBatchSampler
from models.denoiser import ProbNet
from models.util import transform
from models.lr_scheduler import get_lr_scheduler
from models.loss import geodesic_loss
from tqdm import tqdm
import yaml
from core.logger import LogTracker
from core.tools import load_checkpoint, save_checkpoint
import logging
from pathlib import Path
from typing import Dict, Callable
from functools import partial

def get_dataloader(train_base_dataset_argv:Dict, 
        val_base_dataset_argv:Dict,
        train_dataloader_argv:Dict, val_dataloader_argv:Dict):
    train_base_dataset = BaseKITTIDataset(**train_base_dataset_argv)
    val_base_dataset = BaseKITTIDataset(**val_base_dataset_argv)
    train_dataloader_argv['batch_sampler'] = KITTIBatchSampler(len(train_base_dataset.kitti_datalist), train_base_dataset.sep, **train_dataloader_argv['batch_sampler'])
    val_dataloader_argv['batch_sampler'] = KITTIBatchSampler(len(val_base_dataset.kitti_datalist), val_base_dataset.sep, **val_dataloader_argv['batch_sampler'])
    if hasattr(train_base_dataset, 'collate_fn'):
        train_dataloader_argv['collate_fn'] = getattr(train_base_dataset, 'collate_fn')
    if hasattr(val_base_dataset, 'collate_fn'):
        val_dataloader_argv['collate_fn'] = getattr(val_base_dataset, 'collate_fn')
    train_dataloader = DataLoader(train_base_dataset, **train_dataloader_argv)
    val_dataloader = DataLoader(val_base_dataset, **val_dataloader_argv)
    return train_dataloader, val_dataloader

@torch.inference_mode()
def val_epoch(val_loader:DataLoader, model:ProbNet, transform_func:Callable, logger:logging.Logger, device, log_per_iter:int):
    model.eval()
    total_loss = 0
    logger.info("Validation:")
    iterator = tqdm(val_loader, desc='val')
    tracker = LogTracker('R','T','log_softmax')
    with iterator:
        N_valid = len(val_loader)
        for i, batch in enumerate(val_loader):
            img = batch['img'].to(device)
            pcd = batch['pcd'].to(device)
            gt_se3 = batch['extran'].to(device)
            B = img.shape[0]
            camera_info = batch['camera_info']
            queries = transform_func().unsqueeze(0).repeat(B, 1, 1, 1).to(gt_se3)  # (B, K, 4, 4)
            queries[:,0,:,:] = gt_se3  # (B, 4, 4)
            probs = model.forward(img, pcd, queries, camera_info, softmax=True)  # maximize the log_probablity of the gt pose
            loss = -torch.log(probs[:,0]).mean()
            best_se3_idx = torch.argmax(probs, dim=1)  # (B, )
            batch_idx = torch.arange(B).to(best_se3_idx)
            best_se3 = queries[batch_idx, best_se3_idx]
            R_loss, t_loss = geodesic_loss(best_se3, gt_se3)
            total_loss += R_loss.item() + t_loss.item()
            tracker.update('R',R_loss.item())
            tracker.update('T',t_loss.item())
            tracker.update('log_softmax',loss.item())
            iterator.set_postfix(tracker.result())
            iterator.update(1)
            if (i+1) % log_per_iter == 0 or (i+1) == len(val_loader):
                logger.info("\tBatch:{}|{}: {}".format(i+1, len(val_loader), tracker.result()))
    assert N_valid > 0, "Fatal Error, no valid batch!"
    return total_loss / N_valid



def main(config:Dict, config_path:str):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = config['device']
    # torch.backends.cudnn.benchmark=True
    # torch.backends.cudnn.enabled = False
    model = ProbNet(**config['model']).to(device)
    dataset_argv = config['dataset']['train']
    train_dataloader, val_dataloader = get_dataloader(dataset_argv['dataset']['train']['base'],
        dataset_argv['dataset']['val']['base'], dataset_argv['dataloader']['args'], dataset_argv['dataloader']['val_args'])
    optimizer = torch.optim.Adam(model.mlp.parameters(), **config['optimizer']['args'])  # only optimize mlp
    clip_grad = config['optimizer']['max_grad']
    scheduler = get_lr_scheduler(optimizer, config['scheduler']['type'], **config['scheduler']['args'])
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
        start_epoch, best_loss = load_checkpoint(path_argv['resume'], model.mlp, optimizer, scheduler)
        logger.info("Loaded checkpoint from {}, Start from Epoch {}".format(path_argv['resume'], start_epoch))
    else:
        start_epoch = 1
        best_loss = float('inf')
        logger.info("Start from scratch")
    num_queries = run_argv['num_queries']
    se3_transform_func = transform.UniformTransformSE3(**dataset_argv['dataset']['train']['main'])
    val_transform_func = transform.UniformTransformSE3(**dataset_argv['dataset']['val']['main'])
    ## training
    for epoch_idx in range(start_epoch, run_argv['n_epoch']+1):
        model.train()
        iterator = tqdm(train_dataloader, desc='train')
        tracker = LogTracker('log_softmax')
        with iterator:
            for i, batch in enumerate(train_dataloader):
                # model prediction
                img = batch['img'].to(device)
                pcd = batch['pcd'].to(device)
                gt_se3 = batch['extran'].to(device)
                camera_info = batch['camera_info']
                B = img.shape[0]
                optimizer.zero_grad()
                queries = se3_transform_func.generate_transform(num_queries, return_se3=True).unsqueeze(0).repeat(B,1,1,1).to(gt_se3)  # (K, 4, 4)
                queries[:,0,:,:] = gt_se3  # (B, 4, 4)
                probs = model.forward(img, pcd, queries, camera_info, softmax=True)  # maximize the log_probablity of the gt pose
                loss = -torch.log(probs[:,0]).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad, norm_type=2)  # avoid gradient explosion
                optimizer.step()
                tracker.update('log_softmax', loss.item())
                iterator.set_postfix(tracker.result())
                iterator.update(1)
                if (i+1) % run_argv['log_per_iter'] == 0:
                    logger.info("\tBatch {}|{}: {}".format(i+1, len(train_dataloader), tracker.result()))
            logger.info("Epoch {}|{}: {}".format(epoch_idx, run_argv['n_epoch'], tracker.result()))
            scheduler.step()
            save_checkpoint(str(checkpoints_dir.joinpath('last_model.pth')), epoch_idx, best_loss, model.mlp, optimizer, scheduler)
        if epoch_idx % run_argv['val_per_epoch'] == 0:
            val_loss = val_epoch(val_dataloader, model, partial(val_transform_func.generate_transform, num=run_argv['val_num_queries'], return_se3=True), logger, device, run_argv['log_per_iter'])
            if val_loss < best_loss:
                logger.info("Find Best Model at Epoch {} prev | curr best loss: {} | {}".format(epoch_idx, best_loss, val_loss))
                best_loss = val_loss
                save_checkpoint(str(checkpoints_dir.joinpath('best_model.pth')), epoch_idx, best_loss, model.mlp, optimizer, scheduler)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="cfg/prob.yml", type=str)
    args = parser.parse_args()
    config = yaml.load(open(args.config,'r'), yaml.SafeLoader)
    main(config, args.config)