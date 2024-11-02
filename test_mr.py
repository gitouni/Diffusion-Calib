import os
import shutil
import numpy as np
import argparse
from torchinfo import summary
import torch
from torch.utils.data import DataLoader
from dataset import PerturbDataset
from dataset import __classdict__ as DatasetDict
from models.denoiser import Denoiser, RAFTDenoiser, Surrogate, __classdict__ as DenoiserDict
from models.diffuser import Diffuser
from models.loss import se3_err, get_loss
from tqdm import tqdm
import yaml
from models.util import se3
from core.logger import LogTracker
from core.tools import load_checkpoint_model_only
import logging
from pathlib import Path
from typing import Dict, Literal, Iterable, List
import time


def get_dataloader(test_dataset_argv:Iterable[Dict], test_dataloader_argv:Dict, dataset_type:str):
    name_list = []
    dataloader_list = []
    data_class = DatasetDict[dataset_type]
    for dataset_argv in test_dataset_argv:
        name_list.append(dataset_argv['name'])
        base_dataset = data_class(**dataset_argv['base'])
        dataset = PerturbDataset(base_dataset, **dataset_argv['main'])
        if hasattr(dataset, 'collate_fn'):
            test_dataloader_argv['collate_fn'] = getattr(dataset, 'collate_fn')
        dataloader = DataLoader(dataset, **test_dataloader_argv)
        dataloader_list.append(dataloader)
    return name_list, dataloader_list

def to_npy(x0:torch.Tensor) -> np.ndarray:
    return x0.detach().cpu().numpy()


@torch.inference_mode()
def test_multirange(test_loader:DataLoader, name:str, model_list:List[Surrogate], logger:logging.Logger,
        device:torch.device, log_per_iter:int, res_dir:Path):
    for model in model_list:
        model.eval()
    logger.info("Test:")
    iterator = tqdm(test_loader, desc=name)
    tracker = LogTracker('Rx','Ry','Rz','tx','ty','tz','R','t','3d3c','5d5c','time')
    cnt = 0
    with iterator:
        N_valid = len(test_loader)
        for i, batch in enumerate(test_loader):
            img = batch['img'].to(device)
            pcd = batch['pcd'].to(device)
            init_extran = batch['extran'].to(device)
            gt_se3 = batch['gt'].to(device)  # transform uncalibrated_pcd to calibrated_pcd
            batch_n = len(gt_se3)
            camera_info = batch['camera_info']
            H0 = torch.eye(4).unsqueeze(0).to(gt_se3)
            x0_list = []
            t0 = time.time()
            for model in model_list:
                delta_x = model.forward(img, pcd, H0 @ init_extran, camera_info)
                if not isinstance(delta_x, torch.Tensor):
                    H0 = delta_x[-1] @ H0
                else:
                    H0 = se3.exp(delta_x) @ H0
                save_x = to_npy(se3.log(H0 @ init_extran))  # (6,)
                x0_list.append(save_x)
            dt = time.time() - t0
            tracker.update('time', dt, batch_n)
            batched_x0_list = np.stack(x0_list, axis=1)  # (B, K, 6)
            for x0 in batched_x0_list:
                np.savetxt(res_dir.joinpath("%06d.txt"%cnt), x0)
                cnt += 1
            model.clear_buffer()
            R_err, t_err = se3_err(H0, gt_se3)
            R_err = torch.rad2deg(R_err)  # log degree
            if torch.isnan(R_err).sum() + torch.isnan(t_err).sum() > 0:
                logger.warning("nan value detected, skip this batch.")
                N_valid -= 1
                continue
            tracker.update('Rx',torch.mean(R_err[:,0].abs()).item(), batch_n)
            tracker.update('Ry',torch.mean(R_err[:,1].abs()).item(), batch_n)
            tracker.update('Rz',torch.mean(R_err[:,2].abs()).item(), batch_n)
            tracker.update('tx',torch.mean(t_err[:,0].abs()).item(), batch_n)
            tracker.update('ty',torch.mean(t_err[:,1].abs()).item(), batch_n)
            tracker.update('tz',torch.mean(t_err[:,2].abs()).item(), batch_n)
            R_rmse = torch.linalg.norm(R_err, dim=1)
            t_rmse = torch.linalg.norm(t_err, dim=1)
            tracker.update('R',R_rmse.mean().item(), batch_n)
            tracker.update('t',t_rmse.mean().item(), batch_n)
            tracker.update('3d3c', torch.sum(torch.logical_and(R_rmse < 3, t_rmse < 0.03)).item() / batch_n, batch_n)
            tracker.update('5d5c', torch.sum(torch.logical_and(R_rmse < 5, t_rmse < 0.05)).item() / batch_n, batch_n)
            iterator.set_postfix(tracker.result())
            iterator.update(1)
            if (i+1) % log_per_iter == 0:
                logger.info("\tBatch:{}|{}: {}".format(i+1, len(test_loader), tracker.result()))
    assert N_valid > 0, "Fatal Error, no valid batch!"
    return tracker.result(), N_valid / len(test_loader)

def main(config:Dict, config_path:str):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = config['device']
    run_argv = config['run']
    path_argv = config['path']
    dataset_argv = config['dataset']['test']
    dataset_type = config['dataset']['type']
    steps = len(config['stages'])
    name = "mr_{}".format(steps)
    experiment_dir = Path(path_argv['base_dir'])
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(path_argv['name'])
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(dataset_type)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath(path_argv['checkpoint'])
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath(path_argv['log'])
    log_dir.mkdir(exist_ok=True)
    shutil.copyfile(config_path, str(log_dir.joinpath(os.path.basename(config_path))))  # copy the config file
    res_dir = experiment_dir.joinpath(path_argv['results']).joinpath(name)
    if res_dir.exists():
        shutil.rmtree(str(res_dir))
    res_dir.mkdir(exist_ok=True,parents=True)
    
    # logger
    logger = logging.getLogger(path_argv['log'])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_mode = 'w'
    file_handler = logging.FileHandler(str(log_dir) + '/test_{}.log'.format(name), mode=logger_mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('start traing')
    logger.info('args:')
    logger.info(args)
    model_list = []
    assert path_argv['pretrain'] is not None, 'pretrained path must be assigned during test time.'
    for pretrained_path in path_argv['pretrain']:
        surrogate_model:Surrogate = DenoiserDict[config['model']['surrogate']['type']](**config['model']['surrogate']['argv']).to(device)
        load_checkpoint_model_only(pretrained_path, surrogate_model)
        model_list.append(surrogate_model)
        logger.info("Loaded checkpoint from {}".format(pretrained_path))
    # summary(surrogate_model)  # print the volume of model parameters
    # exit(0)
    name_list, dataloader_list = get_dataloader(dataset_argv['dataset'], dataset_argv['dataloader'], dataset_type)
    # testing
    record_list = []
    for name, dataloader in zip(name_list, dataloader_list):
        sub_res_dir = res_dir.joinpath(name)
        sub_res_dir.mkdir()
        record, valid_ratio = test_multirange(dataloader,name, model_list, logger, device, run_argv['log_per_iter'], sub_res_dir)
        logger.info("{}: {} | valid: {:.2%}".format(name, record, valid_ratio))
        record_list.append([name, record, valid_ratio])
    logger.info("Summary:")  # view in the bottom
    for name, record, valid_ratio in record_list:
        logger.info("{}: {} | valid: {:.2%}".format(name, record, valid_ratio))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, default="cfg/dataset/kitti_large.yml")
    parser.add_argument("--model_config",type=str, default="cfg/multirange_model/main_ponly.yml")
    parser.add_argument("--multirange_config",type=str, default="cfg/dataset/multirange.yml")
    args = parser.parse_args()
    dataset_config = yaml.load(open(args.dataset_config,'r'), yaml.SafeLoader)
    multirange_config = yaml.load(open(args.multirange_config, 'r'), yaml.SafeLoader)
    config = yaml.load(open(args.model_config,'r'), yaml.SafeLoader)
    config.update(multirange_config)
    config.update(dataset_config)
    main(config, args.model_config)