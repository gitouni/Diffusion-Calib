import numpy as np
from matplotlib import pyplot as plt
import argparse
from PIL import Image
from models.util.nptrans import toMatw
from models.util.transform import inv_pose_np
from pathlib import Path
from typing import Tuple
from scipy.spatial.transform import Rotation
import shutil
from collections import defaultdict

def se3_err(pred_se3:np.ndarray, gt_se3:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    delta_se3 = pred_se3 @ inv_pose_np(gt_se3)
    delta_euler = np.abs(Rotation.from_matrix(delta_se3[...,:3,:3]).as_euler(seq='XYZ',degrees=True))  # (B, 3)
    delta_tsl = np.abs(delta_se3[...,:3,3])  # (B, 3)
    return delta_euler, delta_tsl  # (B, 3), (B, 3)

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",type=str,default="data/kitti/sequences/13/image_2")
    parser.add_argument("--lidar_dir",type=str,default="data/kitti/sequences/13/velodyne")
    parser.add_argument("--gt_file",type=str,default="cache/kitti_gt/13_gt.txt")
    parser.add_argument("--x0_files",type=str,nargs="+", default=["experiments/kitti/naiter/lccnet/results/iterative_10_2025-02-02-09-23-46/seq_13/000223.txt",
                                                                 "experiments/kitti/nlsd/lccnet/results/nlsd_10_2025-02-02-10-24-47/seq_13/000223.txt",
                                                                 "experiments/kitti/lsd/lccnet/results/unipc_10_2025-02-02-08-06-59/seq_13/000223.txt"
                                                                 ])
    parser.add_argument("--legend",type=str,nargs="+",default=['naiter','nlsd','lsd'])
    parser.add_argument("--axis_names",type=str,nargs="+",default=['Rx','Ry','Rz','tx, ty, tz'])
    parser.add_argument("--axis_unit",type=str,nargs="+",default=['$^\circ$','$^\circ$','$^\circ$','cm','cm','cm'])
    parser.add_argument("--res_dir",type=str,default="fig/err_curve_noaxis")
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    res_dir = Path(args.res_dir)
    if res_dir.exists():
        shutil.rmtree(str(res_dir))
    res_dir.mkdir(parents=True)
    err_dict = defaultdict(lambda: defaultdict(list))
    gt_se3 = np.loadtxt(args.gt_file)
    x0_list = [np.loadtxt(x0_file) for x0_file in args.x0_files]
    for x0, name in zip(x0_list, args.legend):
        for iter_x0 in x0:
            R_err_i, t_err_i = se3_err(toMatw(iter_x0), gt_se3)
            err_dict['Rx'][name].append(R_err_i[0])
            err_dict['Ry'][name].append(R_err_i[1])
            err_dict['Rz'][name].append(R_err_i[2])
            err_dict['tx'][name].append(t_err_i[0] * 100)
            err_dict['ty'][name].append(t_err_i[1] * 100)
            err_dict['tz'][name].append(t_err_i[2] * 100)
    font = {'family' : 'DejaVu Sans',
        'size'   : 22}
    legend_font = {'family' : 'DejaVu Sans',
        'size'   : 22}
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 22
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (ax, axis_name, axis_unit) in enumerate(zip(axes.flat, err_dict.keys(), args.axis_unit)):
        ax:plt.Axes
        xticks = list(range(11))
        for sub_name in args.legend:
            ax.plot(xticks, err_dict[axis_name][sub_name], label=sub_name)
        ax.set_xlim(left=0,right=10)
        ax.set_ylim(bottom=0)
        # plt.xlabel('iterations', fontdict=font)
        # plt.ylabel('{} ({})'.format(axis_name, axis_unit),fontdict=font)
        ax.set_title(f'{axis_name} ({axis_unit})')
        ax.grid(True)
        # legned_loc = 'upper right' if axis_name != 'tx' else 'upper left'
        # plt.legend(args.legend, prop=legend_font, loc=legned_loc)
        # plt.savefig(str(res_dir.joinpath(axis_name + '.png')))  # no padding
    # plt.tight_layout()
    handles, labels = axes[-1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 给底部图例留出空间
    plt.show()