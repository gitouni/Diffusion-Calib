import os
from sploss import CMSCLoss, FLowLoss
from models.tools.cmsc import toMat, toVec, inv_pose
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from core.logger import LogTracker
from typing import Tuple
from scipy.spatial.transform import Rotation
from functools import partial
from geatpy_utils import ea_optimize

def se3_err(pred_se3:np.ndarray, gt_se3:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    delta_se3 = pred_se3 @ inv_pose(gt_se3)
    delta_euler = Rotation.from_matrix(delta_se3[...,:3,:3]).as_euler(seq='xyz', degrees=False)
    delta_tsl = np.abs(delta_se3[...,:3,3])
    return delta_euler, delta_tsl  # (B, 3), (B, 3)

if __name__ == "__main__":
    config = yaml.load(open('cfg/flow_opt.yml','r'), yaml.SafeLoader)
    dataset_cfg_list = config['dataset']
    optimzier_cfg = config['optimizer']
    
    for dataset_cfg in dataset_cfg_list:
        LossClass = CMSCLoss if dataset_cfg['loss']['type'] == 'cmsc' else FLowLoss
        name = dataset_cfg['name']
        Loss = LossClass(**dataset_cfg['loss']['argv'])
        # multiproc_fn = lambda x_list: np.array(Parallel(n_jobs=-1)(delayed(loss_fn)(x) for x in x_list.T))
        gt_extran = np.loadtxt(dataset_cfg['gt_file'])
        # perturb_se3 = np.loadtxt(dataset_cfg['perturb_file'])
        init_files = sorted(os.listdir(dataset_cfg['pred_dir']))
        save_path = Path(dataset_cfg['res_dir'])
        save_path.mkdir(exist_ok=True, parents=True)
        titer = tqdm(total=len(init_files), desc=name)
        tracker = LogTracker('Rx','Ry','Rz','tx','ty','tz','R','t')
        with titer:
            for i, init_file in enumerate(init_files):
                init_se3 = np.loadtxt(os.path.join(dataset_cfg['pred_dir'], init_file))
                if np.ndim(init_se3) == 2:
                    init_se3 = init_se3[-1]  # the last iteration
                init_extran = toMat(init_se3[:3], init_se3[3:])
                x0 = np.zeros(6)
                x0[:3], x0[3:] = Loss.toVec(init_extran)
                lb = np.array(optimzier_cfg['min_bnd']) + x0
                ub = np.array(optimzier_cfg['max_bnd']) + x0
                if dataset_cfg['loss']['argv']['place_holder'] is None:
                    if dataset_cfg['loss']['type'] == 'cmsc':
                        loss_fn = partial(Loss.weighted_loss, weights=optimzier_cfg['weights'], x0_ref=None)
                    else:
                        loss_fn = partial(Loss, x0_ref=x0_ref)
                else:
                    x0_ref = x0.copy()
                    x0_rev = np.array(dataset_cfg['loss']['argv']['place_holder'],dtype=np.bool_)
                    x0 = x0[x0_rev]
                    lb = lb[x0_rev]
                    ub = ub[x0_rev]
                    if dataset_cfg['loss']['type'] == 'cmsc':
                        loss_fn = partial(Loss.weighted_loss, weights=optimzier_cfg['weights'], x0_ref=x0_ref)
                    else:
                        loss_fn = partial(Loss, x0_ref=x0_ref)
                # y0 = loss_fn(x0)
                res = ea_optimize(lb, ub, x0, 'local', loss_fn, None, NIND=30, MAXGEN=5, requires_multiproc=True, logTras=None)
                if dataset_cfg['loss']['argv']['place_holder'] is None:
                    xopt = res['Vars'].flatten()
                else:
                    xopt = x0_ref.copy()
                    xopt[x0_rev] = res['Vars'].flatten()
                xopt_extran = Loss.toMat(xopt[:3],xopt[3:])
                xopt[:3], xopt[3:] = toVec(xopt_extran)
                np.savetxt(save_path.joinpath(init_file), xopt)
                # xopt = np.loadtxt(save_path.joinpath(init_file))
                R_err, t_err = se3_err(xopt_extran, gt_extran)
                tracker.update('Rx',R_err[0].item())
                tracker.update('Ry',R_err[1].item())
                tracker.update('Rz',R_err[2].item())
                tracker.update('tx',t_err[0].item())
                tracker.update('ty',t_err[1].item())
                tracker.update('tz',t_err[2].item())
                tracker.update('R',np.linalg.norm(R_err).item())
                tracker.update('t',np.linalg.norm(t_err).item())
                titer.set_postfix(tracker.result())
                titer.update(1)
