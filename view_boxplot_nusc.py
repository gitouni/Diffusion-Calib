import os
import numpy as np
from models.util.nptrans import toMatw
from scipy.spatial.transform import Rotation
import argparse
from typing import Tuple
from models.util.transform import inv_pose_np
import shutil
from matplotlib import pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

pred_dirs = [
    [
        'experiments/nuscenes/naiter/calibnet/results/iterative_1_2025-06-23-17-09-40',
        'experiments/nuscenes/naiter/rggnet/results/iterative_1_2025-06-23-17-14-54',
        'experiments/nuscenes/naiter/lccnet/results/iterative_1_2025-06-23-17-12-09',
        'experiments/nuscenes/naiter/lccraft_large/results/iterative_1_2025-06-23-17-17-30',
    ],
    [
        'experiments/nuscenes/naiter/calibnet/results/iterative_10_2025-06-23-17-20-31',
        'experiments/nuscenes/naiter/rggnet/results/iterative_10_2025-06-23-17-27-08',
        'experiments/nuscenes/naiter/lccnet/results/iterative_10_2025-06-23-17-23-36',
        'experiments/nuscenes/naiter/lccraft_large/results/iterative_10_2025-06-23-17-30-14',
    ],
    [
        'experiments/nuscenes/nlsd/calibnet/results/nlsd_10_2025-06-23-00-41-17',
        'experiments/nuscenes/nlsd/rggnet/results/nlsd_10_2025-06-23-00-48-40',
        'experiments/nuscenes/nlsd/lccnet/results/nlsd_10_2025-06-23-00-44-35',
        'experiments/nuscenes/nlsd/lccraft_large/results/nlsd_10_2025-06-23-00-52-21',
    ],
    [
        'experiments/nuscenes/lsd/calibnet/results/unipc_10_2025-06-23-00-18-40',
        'experiments/nuscenes/lsd/rggnet/results/unipc_10_2025-06-23-00-26-04',
        'experiments/nuscenes/lsd/lccnet/results/unipc_10_2025-06-23-00-21-56',
        'experiments/nuscenes/lsd/lccraft_large/results/unipc_10_2025-06-23-00-29-43',
    ],
    
]

custom_color_html = [
    '#75C298', '#A0ADD0', '#E47178', '#F5DC75'
]
def hex_to_rgb(hex_color):
    """将HTML颜色的16进制字符串转换为归一化的RGB值0-1"""
    hex_color = hex_color.lstrip('#')  # 去掉前导 #
    
    # 处理缩写形式 #RGB
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])

    if len(hex_color) != 6:
        raise ValueError("Invalid hex color format. Expected '#RRGGBB' or '#RGB'.")

    # 解析 R、G、B，并归一化到 [0,1]
    r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    
    return (r, g, b)

def se3_err(pred_se3:np.ndarray, gt_se3:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    delta_se3 = pred_se3 @ inv_pose_np(gt_se3)
    delta_euler = np.abs(Rotation.from_matrix(delta_se3[...,:3,:3]).as_euler(seq='XYZ',degrees=True))  # (B, 3)
    delta_tsl = np.abs(delta_se3[...,:3,3])  # (B, 3)
    return delta_euler, delta_tsl  # (B, 3), (B, 3)

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_list", type=str, nargs="+", default=['CalibNet','RGGNet','LCCNet','LCCRAFT'])
    parser.add_argument("--iterative_list",type=str, nargs="+",default=['Single','NaIter','NLSD','LSD'])
    parser.add_argument("--gt_dir",type=str,default="cache/nuscenes_gt")
    parser.add_argument("--res_dir",type=str,default="fig/boxplot/nusc")
    parser.add_argument("--sample_num",type=int,default=300)
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    custom_colors = list(map(hex_to_rgb, custom_color_html))  # list of [r,g,b]
    gt_files = sorted(os.listdir(args.gt_dir))
    n_method, n_iterative = len(pred_dirs[0]), len(pred_dirs)
    assert n_method == len(args.method_list)
    assert n_iterative == len(args.iterative_list)
    err_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # Rx -> calibnet -> lsd -> []
    # Q75_dict = defaultdict(float)
    progress = tqdm(desc='get data',total=n_iterative * n_method)
    with progress:
        for iter_method, iter_pred_dirs in zip(args.iterative_list, pred_dirs):
            for method, pred_dir in zip(args.method_list, iter_pred_dirs):
                progress.set_description('iter:{}, method:{}'.format(iter_method, method))
                sub_pred_dirs = sorted(os.listdir(pred_dir))
                assert len(gt_files) == len(sub_pred_dirs), "number of gt files ({}) != number of pred subdirs ({})".format(len(gt_files), len(sub_pred_dirs))
                for gt_file, pred_subdir in zip(gt_files, sub_pred_dirs):
                    gt_se3 = np.loadtxt(os.path.join(args.gt_dir, gt_file))
                    pred_full_dir = os.path.join(pred_dir, pred_subdir)
                    pred_files = sorted(os.listdir(pred_full_dir))
                    R_err = np.zeros([len(pred_files), 3])
                    t_err = np.zeros([len(pred_files), 3])
                    for i, pred_file in enumerate(pred_files):
                        pred_se3_i = np.loadtxt(os.path.join(pred_full_dir, pred_file))
                        if np.ndim(pred_se3_i) == 2:
                            pred_se3_i = pred_se3_i[-1]  # sequences of prediction
                        R_err_i, t_err_i = se3_err(toMatw(pred_se3_i), gt_se3)
                        R_err[i, :] = R_err_i
                        t_err[i, :] = t_err_i
                    RRMSE:np.ndarray = np.sqrt(np.sum(R_err**2, axis=1))
                    TRMSE:np.ndarray = np.sqrt(np.sum(t_err**2, axis=1))
                    dataset_len = len(RRMSE)
                    data_index = np.linspace(0, dataset_len-1, num=args.sample_num, dtype=np.int32)
                    err_dict['Rx'][method][iter_method].extend(R_err[data_index,0].tolist())
                    err_dict['Ry'][method][iter_method].extend(R_err[data_index,1].tolist())
                    err_dict['Rz'][method][iter_method].extend(R_err[data_index,2].tolist())
                    err_dict['tx'][method][iter_method].extend(t_err[data_index,0].tolist())
                    err_dict['ty'][method][iter_method].extend(t_err[data_index,1].tolist())
                    err_dict['tz'][method][iter_method].extend(t_err[data_index,2].tolist())
                    err_dict['RRMSE'][method][iter_method].extend(RRMSE[data_index].tolist())
                    err_dict['TRMSE'][method][iter_method].extend(TRMSE[data_index].tolist())
                    # Q75_dict['Rx'] = max(Q75_dict['Rx'], np.quantile(R_err[data_index,0], 0.75))
                    # Q75_dict['Ry'] = max(Q75_dict['Ry'], np.quantile(R_err[data_index,1], 0.75))
                    # Q75_dict['Rz'] = max(Q75_dict['Rz'], np.quantile(R_err[data_index,2], 0.75))
                    # Q75_dict['tx'] = max(Q75_dict['tx'], np.quantile(t_err[data_index,0], 0.75))
                    # Q75_dict['ty'] = max(Q75_dict['ty'], np.quantile(t_err[data_index,1], 0.75))
                    # Q75_dict['tz'] = max(Q75_dict['tz'], np.quantile(t_err[data_index,2], 0.75))
                    # Q75_dict['RRMSE'] = max(Q75_dict['RRMSE'], np.quantile(RRMSE[data_index], 0.75))
                    # Q75_dict['TRMSE'] = max(Q75_dict['TRMSE'], np.quantile(TRMSE[data_index], 0.75))
                progress.update(1)
    print(err_dict.keys())
    res_path = Path(args.res_dir)
    if res_path.exists():
        shutil.rmtree(str(res_path))
    res_path.mkdir(parents=True)
    # font = {'family' : 'DejaVu Sans',
    #         'size'   : 18}
    # legend_font = {'family' : 'DejaVu Sans',
    #                 'size'   : 18}
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 22

    # # 绘制箱线图
    xticklabels = args.method_list
    for err_name, err_item in err_dict.items():
        
        # 创建绘图
        fig, ax = plt.subplots(figsize=(14, 6))

        # 计算箱线图位置，每组3个箱线图
        positions = np.arange(n_method) * (n_iterative + 1)  # 组间留空

        # 遍历每个误差类别
        for group_i, iter_method in zip(range(n_iterative), args.iterative_list):
            box = ax.boxplot([err_item[method][iter_method] for method in args.method_list],  # 转置以适应 boxplot 需要的数据格式
                            positions=positions + group_i,  # 偏移位置
                            patch_artist=True, widths=0.6, showfliers=False,
                            # whiskerprops={'visible': False},  # 隐藏须线
                            # capprops={'visible': False},  # 隐藏端点
                            medianprops={'color': '#383838', 'linewidth': 2})  # 中位数线的颜色和粗细)

            # 设置颜色
            for patch in box['boxes']:
                patch.set_facecolor(custom_colors[group_i])  # 使用自定义 RGB 颜色

        # 设置 x 轴刻度和标签
        ax.set_xticks(positions + (n_iterative - 1) / 2)  # 让刻度对齐到组的中心
        ax.set_xticklabels(xticklabels)
        # ax.set_ylim(0, Q75_dict[err_name] * 1.05)
        # 添加图例
        handles = [plt.Line2D([0], [0], color=color, lw=4) for color in custom_colors]
        # handles.append(plt.Line2D([], [], color='#383838', linewidth=2, linestyle='-')) # 中位数
        # handles.append(plt.Line2D([], [], color='#383838', linewidth=2, linestyle=':'))    # 均值)
        ax.legend(handles, args.iterative_list, loc='upper right',ncol=4)
        
        plt.savefig(str(res_path.joinpath('{}.pdf'.format(err_name))), bbox_inches='tight')
        plt.close()