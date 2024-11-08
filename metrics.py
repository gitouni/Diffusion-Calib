import os
import numpy as np
from models.util.nptrans import toMatw
from scipy.spatial.transform import Rotation
import argparse
from typing import Tuple
from models.util.transform import inv_pose_np
from pprint import pprint
from collections import OrderedDict
import json
from pathlib import Path

def se3_err(pred_se3:np.ndarray, gt_se3:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    delta_se3 = pred_se3 @ inv_pose_np(gt_se3)
    delta_euler = np.abs(Rotation.from_matrix(delta_se3[...,:3,:3]).as_euler(seq='XYZ',degrees=True))  # (B, 3)
    delta_tsl = np.abs(delta_se3[...,:3,3])  # (B, 3)
    return delta_euler, delta_tsl  # (B, 3), (B, 3)

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir_root",type=str,default="experiments/multirange/main/kitti/results/iterative_1")
    parser.add_argument("--gt_dir",type=str,default="cache/kitti_gt")
    parser.add_argument("--log_file",type=str,default="log/multirange/main.json")
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    gt_files = sorted(os.listdir(args.gt_dir))
    pred_dirs = sorted(os.listdir(args.pred_dir_root))
    assert len(gt_files) == len(pred_dirs), "number of gt files ({}) != number of pred subdirs ({})".format(len(gt_files), len(pred_dirs))
    names = pred_dirs
    metrics = OrderedDict({"Rx":[], "Ry":[], "Rz":[], "tx":[], "ty":[], "tz":[],"R":[],"t":[],"3d3c":[],"5d5c":[]})
    print("Compute metrics on {}".format(names))
    metric_list = []
    log_path = os.path.dirname(args.log_file)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    if os.path.exists(args.log_file):
        os.remove(args.log_file)
    for name, gt_file, pred_subdir in zip(names, gt_files, pred_dirs):
        gt_se3 = np.loadtxt(os.path.join(args.gt_dir, gt_file))
        pred_dir = os.path.join(args.pred_dir_root, pred_subdir)
        pred_files = sorted(os.listdir(pred_dir))
        R_err = np.zeros([len(pred_files), 3])
        t_err = np.zeros([len(pred_files), 3])
        for i, pred_file in enumerate(pred_files):
            pred_se3_i = np.loadtxt(os.path.join(pred_dir, pred_file))
            if np.ndim(pred_se3_i) == 2:
                pred_se3_i = pred_se3_i[-1]  # sequences of prediction
            R_err_i, t_err_i = se3_err(toMatw(pred_se3_i), gt_se3)
            R_err[i, :] = R_err_i
            t_err[i, :] = t_err_i
        dir_metric = OrderedDict(name=name)
        dir_metric['Rx'] = np.mean(R_err[:,0])
        dir_metric['Ry'] = np.mean(R_err[:,1])
        dir_metric['Rz'] = np.mean(R_err[:,2])
        dir_metric['tx'] = np.mean(t_err[:,0])
        dir_metric['ty'] = np.mean(t_err[:,1])
        dir_metric['tz'] = np.mean(t_err[:,2])
        R_rmse = np.linalg.norm(R_err, axis=1)
        t_rmse = np.linalg.norm(t_err, axis=1)
        dir_metric['R'] = np.mean(R_rmse)
        dir_metric['t'] = np.mean(t_rmse)
        dir_metric['3d3c'] = np.sum(np.logical_and(R_rmse < 3, t_rmse < 0.03)) / len(R_rmse)
        dir_metric['5d5c'] = np.sum(np.logical_and(R_rmse < 5, t_rmse < 0.05)) / len(R_rmse)
        metric_list.append(dir_metric)
        for metric in metrics.keys():
            metrics[metric].append(dir_metric[metric])
    for metric in metrics.keys():
        metrics[metric] = sum(metrics[metric]) / len(metrics[metric])
    metric_list.append(metrics)
    json.dump(metric_list, open(args.log_file,'w'),indent=2)
    print('log file saved to {}'.format(args.log_file))