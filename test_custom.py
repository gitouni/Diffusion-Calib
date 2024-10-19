import os
import shutil
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset
from dataset import CBADataset, PerturbDataset
from models.denoiser import Surrogate, Denoiser
from models.diffuser import Diffuser
from models.loss import se3_err, get_loss
from tqdm import tqdm
import yaml
from models.util import se3
from core.logger import LogTracker
from core.tools import load_checkpoint_model_only
import logging
from pathlib import Path
from typing import Dict, Literal, Iterable, Optional


def get_dataset(test_dataset_argv:Iterable[Dict]):
    name_list = []
    dataset_list = []
    for dataset_argv in test_dataset_argv:
        name_list.append(dataset_argv['name'])
        base_dataset = CBADataset(**dataset_argv['base'])
        dataset = PerturbDataset(base_dataset, **dataset_argv['main'])
        dataset_list.append(dataset)
    return name_list, dataset_list

def to_npy(x0:torch.Tensor) -> np.ndarray:
    return x0.squeeze(0).detach().cpu().numpy()

@torch.inference_mode()
def test_diffuser(test_dataset:Dataset, name:str, diffuser:Diffuser, logger:logging.Logger, device:torch.device, log_per_iter:int, res_dir:Path):
    diffuser.x0_fn.model.eval()
    logger.info("Test:")
    iterator = tqdm(test_dataset, desc=name)
    tracker = LogTracker('Rx','Ry','Rz','tx','ty','tz','R','t')
    with iterator:
        N_valid = len(test_dataset)
        for i, batch in enumerate(test_dataset):
            img = batch['img'].to(device).unsqueeze(0)
            pcd = batch['pcd'].to(device).unsqueeze(0)
            init_extran = batch['extran'].to(device).unsqueeze(0)
            gt_se3 = batch['gt'].to(device).unsqueeze(0)  # transform uncalibrated_pcd to calibrated_pcd
            gt_x = se3.log(gt_se3)
            camera_info = batch['camera_info']
            x0_hat, x0_list = diffuser.dpm_sampling(torch.zeros_like(gt_x), (img, pcd, init_extran, camera_info), return_intermediate=True)
            x0_list = [to_npy(se3.log(se3.exp(x0) @ init_extran)) for x0 in x0_list]
            np.savetxt(res_dir.joinpath("%06d.txt"%i), np.array(x0_list))
            x0_se3 = se3.exp(x0_hat)
            R_err, t_err = se3_err(x0_se3, gt_se3)
            if torch.isnan(R_err).sum() + torch.isnan(t_err).sum() > 0:
                logger.warn("nan value detected, skip this batch.")
                N_valid -= 1
                continue
            batch_n = len(gt_se3)
            tracker.update('Rx',torch.mean(R_err[:,0].abs()).item(), batch_n)
            tracker.update('Ry',torch.mean(R_err[:,1].abs()).item(), batch_n)
            tracker.update('Rz',torch.mean(R_err[:,2].abs()).item(), batch_n)
            tracker.update('tx',torch.mean(t_err[:,0].abs()).item(), batch_n)
            tracker.update('ty',torch.mean(t_err[:,1].abs()).item(), batch_n)
            tracker.update('tz',torch.mean(t_err[:,2].abs()).item(), batch_n)
            tracker.update('R',torch.linalg.norm(R_err, dim=1).mean().item(), batch_n)
            tracker.update('t',torch.linalg.norm(t_err, dim=1).mean().item(), batch_n)
            iterator.set_postfix(tracker.result())
            iterator.update(1)
            if (i+1) % log_per_iter == 0:
                logger.info("\tBatch:{}|{}: {}".format(i+1, len(test_dataset), tracker.result()))
    assert N_valid > 0, "Fatal Error, no valid batch!"
    return tracker.result(), N_valid / len(test_dataset)

@torch.no_grad()
def test_diffuser_with_guidance(test_dataset:Dataset, name:str, diffuser:Diffuser, logger:logging.Logger, device:torch.device, log_per_iter:int, res_dir:Path,
        classifier_fn_argv:Dict, guidance_scale:float, classifier_t_threshold:float, classifier_grad_place_holder:Optional[Iterable]=None):
    diffuser.x0_fn.model.eval()
    logger.info("Test:")
    iterator = tqdm(test_dataset, desc=name)
    tracker = LogTracker('Rx','Ry','Rz','tx','ty','tz','R','t')
    with iterator:
        N_valid = len(test_dataset)
        for i, batch in enumerate(test_dataset):
            img = batch['img'].to(device).unsqueeze(0)
            pcd = batch['pcd'].to(device).unsqueeze(0)
            init_extran = batch['extran'].to(device).unsqueeze(0)
            gt_se3 = batch['gt'].to(device).unsqueeze(0)  # transform uncalibrated_pcd to calibrated_pcd
            gt_x = se3.log(gt_se3)
            camera_info = batch['camera_info']
            x0_hat, x0_list = diffuser.dpm_sampling_with_guidance(torch.zeros_like(gt_x), (img, pcd, init_extran, camera_info), batch['cba_data'],  batch['ca_data'],
                classifier_fn_argv, guidance_scale, classifier_t_threshold, classifier_grad_place_holder, return_intermediate=True)
            x0_se3 = se3.exp(x0_hat)
            x0_list = [to_npy(se3.log(se3.exp(x0) @ init_extran)) for x0 in x0_list]
            np.savetxt(res_dir.joinpath("%06d.txt"%i), np.array(x0_list))
            R_err, t_err = se3_err(x0_se3, gt_se3)
            if torch.isnan(R_err).sum() + torch.isnan(t_err).sum() > 0:
                logger.warn("nan value detected, skip this batch.")
                N_valid -= 1
                continue
            batch_n = len(gt_se3)
            tracker.update('Rx',torch.mean(R_err[:,0].abs()).item(), batch_n)
            tracker.update('Ry',torch.mean(R_err[:,1].abs()).item(), batch_n)
            tracker.update('Rz',torch.mean(R_err[:,2].abs()).item(), batch_n)
            tracker.update('tx',torch.mean(t_err[:,0].abs()).item(), batch_n)
            tracker.update('ty',torch.mean(t_err[:,1].abs()).item(), batch_n)
            tracker.update('tz',torch.mean(t_err[:,2].abs()).item(), batch_n)
            tracker.update('R',torch.linalg.norm(R_err, dim=1).mean().item(), batch_n)
            tracker.update('t',torch.linalg.norm(t_err, dim=1).mean().item(), batch_n)
            iterator.set_postfix(tracker.result())
            iterator.update(1)
            if (i+1) % log_per_iter == 0:
                logger.info("\tBatch:{}|{}: {}".format(i+1, len(test_dataset), tracker.result()))
    assert N_valid > 0, "Fatal Error, no valid batch!"
    return tracker.result(), N_valid / len(test_dataset)

@torch.inference_mode()
def test_iterative(test_dataset:Dataset, name:str, model:Surrogate, logger:logging.Logger, device:torch.device, log_per_iter:int, res_dir:Path, iters:int):
    model.eval()
    logger.info("Test:")
    iterator = tqdm(test_dataset, desc=name)
    tracker = LogTracker('Rx','Ry','Rz','tx','ty','tz','R','t')
    with iterator:
        N_valid = len(test_dataset)
        for i, batch in enumerate(test_dataset):
            img = batch['img'].to(device).unsqueeze(0)
            pcd = batch['pcd'].to(device).unsqueeze(0)
            init_extran = batch['extran'].to(device).unsqueeze(0)
            gt_se3 = batch['gt'].to(device).unsqueeze(0)  # transform uncalibrated_pcd to calibrated_pcd
            camera_info = batch['camera_info']
            H0 = torch.eye(4).unsqueeze(0).to(gt_se3)
            model.restore_buffer(img, pcd)
            x0_list = []
            for _ in range(iters):
                pcd_tf = se3.transform(H0, pcd)
                delta_x = model.forward(img, pcd_tf, init_extran, camera_info)
                H0 = se3.exp(delta_x) @ H0
                save_x = to_npy(se3.log(H0 @ init_extran))  # (6,)
                x0_list.append(save_x)
            np.savetxt(res_dir.joinpath("%06d.txt"%i), np.array(x0_list))
            model.clear_buffer()
            R_err, t_err = se3_err(H0, gt_se3)
            if torch.isnan(R_err).sum() + torch.isnan(t_err).sum() > 0:
                logger.warn("nan value detected, skip this batch.")
                N_valid -= 1
                continue
            batch_n = len(gt_se3)
            tracker.update('Rx',torch.mean(R_err[:,0].abs()).item(), batch_n)
            tracker.update('Ry',torch.mean(R_err[:,1].abs()).item(), batch_n)
            tracker.update('Rz',torch.mean(R_err[:,2].abs()).item(), batch_n)
            tracker.update('tx',torch.mean(t_err[:,0].abs()).item(), batch_n)
            tracker.update('ty',torch.mean(t_err[:,1].abs()).item(), batch_n)
            tracker.update('tz',torch.mean(t_err[:,2].abs()).item(), batch_n)
            tracker.update('R',torch.linalg.norm(R_err, dim=1).mean().item(), batch_n)
            tracker.update('t',torch.linalg.norm(t_err, dim=1).mean().item(), batch_n)
            iterator.set_postfix(tracker.result())
            iterator.update(1)
            if (i+1) % log_per_iter == 0:
                logger.info("\tBatch:{}|{}: {}".format(i+1, len(test_dataset), tracker.result()))
    assert N_valid > 0, "Fatal Error, no valid batch!"
    return tracker.result(), N_valid / len(test_dataset)

def main(config:Dict, config_path:str, model_type:Literal['diffusion','iterative','diffusion-guide'], iters:int):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = config['device']
    dataset_argv = config['dataset']['args']
    name_list, dataset_list = get_dataset(dataset_argv)
    # torch.backends.cudnn.benchmark=True
    # torch.backends.cudnn.enabled = False
    surrogate_model = Surrogate(**config['model']['surrogate']).to(device)
    
    run_argv = config['run']
    path_argv = config['path']
    experiment_dir = Path(path_argv['base_dir'])
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(path_argv['name'])
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath(path_argv['checkpoint'])
    checkpoints_dir.mkdir(exist_ok=True)
    result_dir = experiment_dir.joinpath(path_argv['results']).joinpath(model_type)
    result_dir.mkdir(exist_ok=True, parents=True)
    log_dir = experiment_dir.joinpath(path_argv['log'])
    log_dir.mkdir(exist_ok=True)
    shutil.copyfile(config_path, str(log_dir.joinpath(os.path.basename(config_path))))  # copy the config file
    # logger
    logger = logging.getLogger(path_argv['log'])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_mode = 'w'
    steps = config['model']['diffuser']['dpm_argv']['steps'] if 'diffusion' in model_type else iters
    file_handler = logging.FileHandler(str(log_dir) + '/test_{}_{}.log'.format(model_type, steps), mode=logger_mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('start traing')
    logger.info('args:')
    logger.info(args)
    if path_argv['pretrain'] is not None:
        load_checkpoint_model_only(path_argv['pretrain'], surrogate_model)
        logger.info("Loaded checkpoint from {}".format(path_argv['pretrain']))
    else:
        raise FileNotFoundError("'pretrain' cannot be set to 'None' during test-time")
    ## training
    record_list = []
    if 'diffusion' in model_type:
        denoiser = Denoiser(surrogate_model)
        diffuser = Diffuser(denoiser, **config['model']['diffuser'])
        loss_func = get_loss(config['loss']['type'], **config['loss']['args'])
        diffuser.set_loss(loss_func)
        diffuser.set_new_noise_schedule(device)
        if model_type == 'diffusion-guide':
            classifier_argv = config['model']['guidance']['argv']
            classifier_argv['loss_fn'] = get_loss(classifier_argv['loss_fn']['type'], **classifier_argv['loss_fn']['args'])
            classifier_t_threshold = config['model']['guidance']['t_threshold']
            classifier_grad_place_holder = config['model']['guidance']['grad_place_holder']
            guidance_scale = config['model']['guidance']['scale']
    for name, dataset in zip(name_list, dataset_list):
        surrogate_model.train()
        res_dir = result_dir.joinpath(name)
        res_dir.mkdir(exist_ok=True, parents=True)
        if model_type == 'diffusion':
            record, valid_ratio = test_diffuser(dataset, name, diffuser, logger, device, run_argv['log_per_iter'], res_dir)
        elif model_type == 'diffusion-guide':
            record, valid_ratio = test_diffuser_with_guidance(dataset, name, diffuser, logger, device, run_argv['log_per_iter'], res_dir, classifier_argv, guidance_scale, classifier_t_threshold, classifier_grad_place_holder)
        elif model_type == 'iterative':
            record, valid_ratio = test_iterative(dataset, name, surrogate_model, logger, device, run_argv['log_per_iter'], res_dir, iters)
        else:
            raise NotImplementedError("Unknown model_type:{}".format(model_type))
        logger.info("{}: {} | valid: {:.2%}".format(name, record, valid_ratio))
        record_list.append([name, record, valid_ratio])
    logger.info("Summary:")  # view in the bottom
    for name, record, valid_ratio in record_list:
        logger.info("{}: {} | valid: {:.2%}".format(name, record, valid_ratio))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="cfg/custom_main.yml", type=str)
    parser.add_argument('--model_type',type=str, choices=['diffusion','iterative','diffusion-guide'], default='diffusion')
    parser.add_argument("--iters",type=int,default=1)
    args = parser.parse_args()
    config = yaml.load(open(args.config,'r'), yaml.SafeLoader)
    main(config, args.config, args.model_type, args.iters)