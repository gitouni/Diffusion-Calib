import os
import numpy as np
from models.util.nptrans import toMatw
from scipy.spatial.transform import Rotation
import argparse
from typing import Tuple
from models.util.transform import inv_pose_np
from pprint import pprint
from collections import OrderedDict

def se3_err(pred_se3:np.ndarray, gt_se3:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    delta_se3 = pred_se3 @ inv_pose_np(gt_se3)
    delta_euler = np.abs(Rotation.from_matrix(delta_se3[...,:3,:3]).as_euler(seq='XYZ',degrees=True))  # (B, 3)
    delta_tsl = np.abs(delta_se3[...,:3,3])  # (B, 3)
    return delta_euler, delta_tsl  # (B, 3), (B, 3)

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir1",type=str,default="experiments/large/main/kitti/results/se3_diffusion_10")
    parser.add_argument("--pred_dir2",type=str,default="experiments/large/main/kitti/results/unipc_10")
    parser.add_argument("--gt_dir",type=str,default="cache/kitti_gt")
    parser.add_argument("--log_file",type=str,default="log/main_diff_cmp.log")
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    gt_files = sorted(os.listdir(args.gt_dir))
    pred_dirs = sorted(os.listdir(args.pred_dir1))  # the same with args.pred_dir2
    assert len(gt_files) == len(pred_dirs), "number of gt files ({}) != number of pred subdirs ({})".format(len(gt_files), len(pred_dirs))
    names = pred_dirs
    metrics = OrderedDict({"Rx":[], "Ry":[], "Rz":[], "tx":[], "ty":[], "tz":[],"R":[],"t":[],"3d3c":[],"5d5c":[]})
    print("Compute metrics on {}".format(names))
    if os.path.exists(args.log_file):
        os.remove(args.log_file)
    for name, gt_file, pred_subdir in zip(names, gt_files, pred_dirs):
        gt_se3 = np.loadtxt(os.path.join(args.gt_dir, gt_file))
        pred_dir1 = os.path.join(args.pred_dir1, pred_subdir)
        pred_dir2 = os.path.join(args.pred_dir2, pred_subdir)
        pred_files1 = sorted(os.listdir(pred_dir1))
        pred_files2 = sorted(os.listdir(pred_dir2))
        for i, (pred_file1, pred_file2) in enumerate(zip(pred_files1, pred_files2)):
            pred_se3_i1 = np.loadtxt(os.path.join(pred_dir1, pred_file1))
            pred_se3_i2 = np.loadtxt(os.path.join(pred_dir2, pred_file2))
            if np.ndim(pred_se3_i1) == 2:
                pred_se3_i1 = pred_se3_i1[-1]  # sequences of prediction
            if np.ndim(pred_se3_i2) == 2:
                pred_se3_i2 = pred_se3_i2[-1]  # sequences of prediction
            R_err_i1, t_err_i1 = se3_err(toMatw(pred_se3_i1), gt_se3)
            R_err_i2, t_err_i2 = se3_err(toMatw(pred_se3_i2), gt_se3)
            if R_err_i1.mean() > 1.3 * R_err_i2.mean() and t_err_i1.mean() > 1.3 * t_err_i2.mean() and t_err_i2.max() < 0.02 and t_err_i1.max() > 0.03:
                print("group:{}, subindex:{}".format(name, i))
            dir_metric = OrderedDict(seq=name, index=i, sd_R=R_err_i1, sd_t=t_err_i1, lsd_R=R_err_i2, lsd_t=t_err_i2)
            pprint(dir_metric, open(args.log_file,'a'))
    print('log file saved to {}'.format(args.log_file))