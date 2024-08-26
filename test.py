import os
import shutil
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import BaseKITTIDataset, PertubKITTIDataset
from models.denoiser import Denoiser, Surrogate, __classdict__ as DenoiserDict
from models.diffuser import Diffuser
from models.loss import se3_err, get_loss
from tqdm import tqdm
import yaml
from models.util import se3
from core.logger import LogTracker
from core.tools import load_checkpoint_model_only
import logging
from pathlib import Path
from typing import Dict, Literal, Iterable


def get_dataloader(test_dataset_argv:Iterable[Dict], test_dataloader_argv:Dict):
    name_list = []
    dataloader_list = []
    for dataset_argv in test_dataset_argv:
        name_list.append(dataset_argv['name'])
        base_dataset = BaseKITTIDataset(**dataset_argv['base'])
        dataset = PertubKITTIDataset(base_dataset, **dataset_argv['main'])
        if hasattr(dataset, 'collate_fn'):
            test_dataloader_argv['collate_fn'] = getattr(dataset, 'collate_fn')
        dataloader = DataLoader(dataset, **test_dataloader_argv)
        dataloader_list.append(dataloader)
    return name_list, dataloader_list

def to_npy(x0:torch.Tensor) -> np.ndarray:
    return x0.detach().cpu().numpy()

@torch.inference_mode()
def test_diffuser(test_loader:DataLoader, name:str, diffuser:Diffuser, logger:logging.Logger, device:torch.device, log_per_iter:int, res_dir:Path):
    diffuser.x0_fn.model.eval()
    logger.info("Test:")
    iterator = tqdm(test_loader, desc=name)
    tracker = LogTracker('Rx','Ry','Rz','tx','ty','tz','R','t')
    cnt = 0
    with iterator:
        N_valid = len(test_loader)
        for i, batch in enumerate(test_loader):
            img = batch['img'].to(device)
            pcd = batch['pcd'].to(device)
            init_extran = batch['extran'].to(device)
            gt_se3 = batch['gt'].to(device)  # transform uncalibrated_pcd to calibrated_pcd
            gt_x = se3.log(gt_se3)
            camera_info = batch['camera_info']
            x0_hat, x0_list = diffuser.dpm_sampling(torch.zeros_like(gt_x), (img, pcd, init_extran, camera_info), return_intermediate=True)
            x0_list = [to_npy(se3.log(se3.exp(x0) @ init_extran)) for x0 in x0_list]
            batched_x0_list = np.stack(x0_list, axis=1)  # (B, K, 6)
            for x0 in batched_x0_list:
                np.savetxt(res_dir.joinpath("%06d.txt"%cnt), x0)
                cnt += 1
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
                logger.info("\tBatch:{}|{}: {}".format(i+1, len(test_loader), tracker.result()))
    assert N_valid > 0, "Fatal Error, no valid batch!"
    return tracker.result(), N_valid / len(test_loader)


# @torch.no_grad()
# def test_diffuser_guidance(test_loader:DataLoader, name:str, diffuser:Diffuser, classifer:ProbNet, classifier_fn_argv:Dict, logger:logging.Logger, device:torch.device, log_per_iter:int, res_dir:Path):
#     diffuser.x0_fn.model.eval()
#     logger.info("Test:")
#     iterator = tqdm(test_loader, desc=name)
#     tracker = LogTracker('Rx','Ry','Rz','tx','ty','tz','R','t')
#     cnt = 0
#     with iterator:
#         N_valid = len(test_loader)
#         for i, batch in enumerate(test_loader):
#             img = batch['img'].to(device)
#             pcd = batch['pcd'].to(device)
#             init_extran = batch['extran'].to(device)
#             gt_se3 = batch['gt'].to(device)  # transform uncalibrated_pcd to calibrated_pcd
#             gt_x = se3.log(gt_se3)
#             camera_info = batch['camera_info']
#             x0_hat, x0_list = diffuser.dpm_sampling_guidance(torch.zeros_like(gt_x), (img, pcd, init_extran, camera_info), classifier_fn_argv, classifer.single_foward, return_intermediate=True)
#             x0_list = [to_npy(se3.log(se3.exp(x0) @ init_extran)) for x0 in x0_list]
#             batched_x0_list = np.stack(x0_list, axis=1)  # (B, K, 6)
#             for x0 in batched_x0_list:
#                 np.savetxt(res_dir.joinpath("%06d.txt"%cnt), x0)
#                 cnt += 1
#             x0_se3 = se3.exp(x0_hat)
#             R_err, t_err = se3_err(x0_se3, gt_se3)
#             if torch.isnan(R_err).sum() + torch.isnan(t_err).sum() > 0:
#                 logger.warn("nan value detected, skip this batch.")
#                 N_valid -= 1
#                 continue
#             batch_n = len(gt_se3)
#             tracker.update('Rx',torch.mean(R_err[:,0].abs()).item(), batch_n)
#             tracker.update('Ry',torch.mean(R_err[:,1].abs()).item(), batch_n)
#             tracker.update('Rz',torch.mean(R_err[:,2].abs()).item(), batch_n)
#             tracker.update('tx',torch.mean(t_err[:,0].abs()).item(), batch_n)
#             tracker.update('ty',torch.mean(t_err[:,1].abs()).item(), batch_n)
#             tracker.update('tz',torch.mean(t_err[:,2].abs()).item(), batch_n)
#             tracker.update('R',torch.linalg.norm(R_err, dim=1).mean().item(), batch_n)
#             tracker.update('t',torch.linalg.norm(t_err, dim=1).mean().item(), batch_n)
#             iterator.set_postfix(tracker.result())
#             iterator.update(1)
#             if (i+1) % log_per_iter == 0:
#                 logger.info("\tBatch:{}|{}: {}".format(i+1, len(test_loader), tracker.result()))
#     assert N_valid > 0, "Fatal Error, no valid batch!"
#     return tracker.result(), N_valid / len(test_loader)

@torch.inference_mode()
def test_iterative(test_loader:DataLoader, name:str, model:Surrogate, logger:logging.Logger, device:torch.device, log_per_iter:int, res_dir:Path, iters:int):
    model.eval()
    logger.info("Test:")
    iterator = tqdm(test_loader, desc=name)
    tracker = LogTracker('Rx','Ry','Rz','tx','ty','tz','R','t')
    cnt = 0
    with iterator:
        N_valid = len(test_loader)
        for i, batch in enumerate(test_loader):
            img = batch['img'].to(device)
            pcd = batch['pcd'].to(device)
            init_extran = batch['extran'].to(device)
            gt_se3 = batch['gt'].to(device)  # transform uncalibrated_pcd to calibrated_pcd
            camera_info = batch['camera_info']
            H0 = torch.eye(4).unsqueeze(0).to(gt_se3)
            model.restore_buffer(img,pcd)
            x0_list = []
            for _ in range(iters):
                pcd_tf = se3.transform(H0, pcd)
                delta_x = model.forward(img, pcd_tf, init_extran, camera_info)
                H0 = se3.exp(delta_x) @ H0
                save_x = to_npy(se3.log(H0 @ init_extran))  # (6,)
                x0_list.append(save_x)
            batched_x0_list = np.stack(x0_list, axis=1)  # (B, K, 6)
            for x0 in batched_x0_list:
                np.savetxt(res_dir.joinpath("%06d.txt"%cnt), x0)
                cnt += 1
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
                logger.info("\tBatch:{}|{}: {}".format(i+1, len(test_loader), tracker.result()))
    assert N_valid > 0, "Fatal Error, no valid batch!"
    return tracker.result(), N_valid / len(test_loader)

def main(config:Dict, config_path:str, model_type:Literal['diffusion','iterative','diffusion-guidance'], iters:int):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = config['device']
    # torch.backends.cudnn.benchmark=True
    # torch.backends.cudnn.enabled = False
    surrogate_model = DenoiserDict[config['model']['surrogate']['type']](**config['model']['surrogate']['argv']).to(device)
    denoiser = Denoiser(surrogate_model)
    dataset_argv = config['dataset']['test']
    name_list, dataloader_list = get_dataloader(dataset_argv['dataset'], dataset_argv['dataloader'])
    if 'diffusion' in model_type:
        diffuser = Diffuser(denoiser, **config['model']['diffuser'])
        loss_func = get_loss(config['loss']['type'], **config['loss']['args'])
        diffuser.set_loss(loss_func)
        diffuser.set_new_noise_schedule(device)
        # if model_type == 'diffusion-guidance':
        #     probnet = ProbNet(**config['model']['probnet']['argv']).to(device)
        #     probnet.mlp.load_state_dict(torch.load(config['model']['probnet']['mlp_pretrained'])['model'])
        #     classifier_fn_argv = config['model']['probnet']['cls_argv']
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
    logger_mode = 'w'
    steps = config['model']['diffuser']['dpm_argv']['steps'] if 'diffusion' in model_type else iters
    name = "{}_{}".format(model_type, steps)
    res_dir = experiment_dir.joinpath(path_argv['results']).joinpath(name)
    res_dir.mkdir(exist_ok=True,parents=True)
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
    for name, dataloader in zip(name_list, dataloader_list):
        surrogate_model.train()
        if model_type == 'diffusion' :
            record, valid_ratio = test_diffuser(dataloader, name, diffuser, logger, device, run_argv['log_per_iter'], res_dir)
        elif model_type == 'iterative':
            record, valid_ratio = test_iterative(dataloader,name, surrogate_model, logger, device, run_argv['log_per_iter'], res_dir, iters)
        # elif model_type == 'diffusion-guidance':
        #     record, valid_ratio = test_diffuser_guidance(dataloader, name, diffuser, probnet, classifier_fn_argv, logger, device, run_argv['log_per_iter'], res_dir)
        else:
            raise NotImplementedError("Unknown model_type:{}".format(model_type))
        logger.info("{}: {} | valid: {:.2%}".format(name, record, valid_ratio))
        record_list.append([name, record, valid_ratio])
    logger.info("Summary:")  # view in the bottom
    for name, record, valid_ratio in record_list:
        logger.info("{}: {} | valid: {:.2%}".format(name, record, valid_ratio))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="cfg/kitti.yml", type=str)
    parser.add_argument('--model_type',type=str, choices=['diffusion','diffusion-guidance','iterative'], default='iterative')
    parser.add_argument("--iters",type=int,default=1)
    args = parser.parse_args()
    config = yaml.load(open(args.config,'r'), yaml.SafeLoader)
    main(config, args.config, args.model_type, args.iters)