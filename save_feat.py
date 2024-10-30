import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import PerturbDataset
from dataset import __classdict__ as DatasetDict
from models.denoiser import Surrogate, __classdict__ as DenoiserDict
import yaml
from core.tools import load_checkpoint_model_only
from pathlib import Path
from typing import Dict, Iterable
from models.util.rotation_conversions import euler_angles_to_matrix

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

def main(config:Dict, debug_dir:str):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = config['device']
    surrogate_model:Surrogate = DenoiserDict[config['model']['surrogate']['type']](**config['model']['surrogate']['argv']).to(device)
    path_argv = config['path']
    load_checkpoint_model_only(path_argv['pretrain'], surrogate_model)
    dataset_argv = config['dataset']['test']
    dataset_type = config['dataset']['type']
    data_class = DatasetDict[dataset_type]
    base_dataset = data_class(**dataset_argv['dataset'][0]['base'])  # retrieve the first dataset
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    batch = base_dataset[0]
    img = batch['img'].to(device).unsqueeze(0).repeat(3,1,1,1)
    pcd = batch['pcd'].to(device).unsqueeze(0).repeat(3,1,1)
    gt_se3 = batch['extran'].to(device).unsqueeze(0).repeat(3,1,1)  # transform uncalibrated_pcd to calibrated_pcd
    camera_info = batch['camera_info']
    perturb = torch.eye(4).unsqueeze(0).repeat(3,1,1).to(device)
    perturb_matrix = euler_angles_to_matrix(torch.deg2rad(torch.eye(3)).to(device), convention='XYZ')
    perturb[:,:3,:3] = perturb_matrix
    init_tran = torch.bmm(perturb, gt_se3)
    # set breakpoint below this line
    y0 = surrogate_model.forward(img, pcd, gt_se3, camera_info)
    # set breakpoint below this line
    y0 = surrogate_model.forward(img, pcd, init_tran, camera_info)  # rotation sensitivity
    perturb = torch.eye(4).unsqueeze(0).repeat(3,1,1).to(device)
    perturb[0,0,-1] = 0.01
    perturb[1,1,-1] = 0.01
    perturb[2,2,-1] = 0.01
    init_tran = torch.bmm(perturb, gt_se3)
    y0 = surrogate_model.forward(img, pcd, init_tran, camera_info)  # rotation sensitivity
    return y0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', default="cfg/dataset/kitti_large.yml", type=str)
    parser.add_argument("--model_config",type=str,default="cfg/unipc_model/main.yml")
    parser.add_argument("--save_dir",type=str,default="debug/tensor")
    args = parser.parse_args()
    dataset_config = yaml.load(open(args.dataset_config,'r'), yaml.SafeLoader)
    config = yaml.load(open(args.model_config,'r'), yaml.SafeLoader)
    config.update(dataset_config)
    main(config, args.save_dir)