import numpy as np
from matplotlib import pyplot as plt
import argparse
from PIL import Image
from models.tools.cmsc import toMat, toVec, npproj
from models.diffuser import make_beta_schedule
from pathlib import Path
from tqdm import tqdm
import os
import open3d as o3d
from functools import partial

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",type=str,default="data/colmap/13/images_col/")
    parser.add_argument("--lidar_dir",type=str,default="data/kitti/sequences/13/velodyne")
    parser.add_argument("--gt_x",type=str,default="data/colmap/13/gt.txt")
    parser.add_argument("--perturb_x",type=str,default="cache/custom/13.txt")
    parser.add_argument("--index",type=int,default=0)
    parser.add_argument("--steps",type=int,default=10)
    parser.add_argument("--intran",type=float,nargs=3, default=[7.188560000000e+02, 6.071928000000e+02, 1.852157000000e+02], help='f, cx, cy')
    parser.add_argument("--res_dir",type=str,default="fig/method")
    return parser.parse_args()

def loadpcd(name:str):
    if name.endswith('.bin'):
        return np.fromfile(name, dtype=np.float32).reshape(-1,4)[:,:3]
    elif name.endswith('.npy'):
        return np.load(name)
    else:
        return NotImplementedError()

def topcd(pcd_arr:np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_arr)
    return pcd

def viz_proj_pcd(proj_pcd:np.ndarray, r:np.ndarray, img:np.ndarray, save_name:str):
    H, W = img.shape[:2]
    u, v = proj_pcd[:,0], proj_pcd[:,1]
    plt.figure(figsize=(12,5),dpi=100,tight_layout=True)
    plt.axis([0,W,H,0])
    plt.imshow(img)
    plt.scatter([u],[v],c=[r],cmap='rainbow_r',alpha=0.5,s=2)
    plt.axis('off')
    plt.savefig(save_name, bbox_inches='tight')  # no padding
    plt.close()


def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

def change_background_to_white(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    return False

def capture_img(vis, path:str):
    image = vis.capture_screen_float_buffer()
    plt.imsave(path,np.array(image))
    return False

if __name__ == "__main__":
    args = options()
    res_dir = Path(args.res_dir)
    res_dir.mkdir(exist_ok=True,parents=True)
    img_files = sorted(os.listdir(args.image_dir))
    lidar_files = sorted(os.listdir(args.lidar_dir))
    image = np.array(Image.open(os.path.join(args.image_dir, img_files[args.index])).convert('RGB'))
    img_hw = image.shape[:2]
    pcd = loadpcd(os.path.join(args.lidar_dir, lidar_files[args.index]))
    pcd_o3d = topcd(pcd)
    pcd_o3d, _ = pcd_o3d.remove_statistical_outlier(10,5.0)
    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("B")] = change_background_to_white
    key_to_callback[ord("S")] = partial(capture_img, path=os.path.join(args.res_dir, "pcd_raw.png"))
    o3d.visualization.draw_geometries_with_key_callbacks([pcd_o3d],key_to_callback)
    sampled_pcd_o3d = pcd_o3d.farthest_point_down_sample(4096)
    key_to_callback[ord("S")] = partial(capture_img, path=os.path.join(args.res_dir, "pcd_4096.png"))
    o3d.visualization.draw_geometries_with_key_callbacks([sampled_pcd_o3d],key_to_callback)
    sampled_pcd_o3d = pcd_o3d.farthest_point_down_sample(2048)
    key_to_callback[ord("S")] = partial(capture_img, path=os.path.join(args.res_dir, "pcd_2048.png"))
    o3d.visualization.draw_geometries_with_key_callbacks([sampled_pcd_o3d],key_to_callback)
    sampled_pcd_o3d = pcd_o3d.farthest_point_down_sample(1024)
    key_to_callback[ord("S")] = partial(capture_img, path=os.path.join(args.res_dir, "pcd_1024.png"))
    o3d.visualization.draw_geometries_with_key_callbacks([sampled_pcd_o3d],key_to_callback)
    sampled_pcd_o3d = pcd_o3d.farthest_point_down_sample(512)
    key_to_callback[ord("S")] = partial(capture_img, path=os.path.join(args.res_dir, "pcd_512.png"))
    o3d.visualization.draw_geometries_with_key_callbacks([sampled_pcd_o3d],key_to_callback)
    # f, cx, cy = args.intran
    # intran = np.array([[f, 0, cx],
    #                   [0,f,cy],
    #                   [0,0,1]])
    # gt_x = np.loadtxt(args.gt_x, dtype=np.float32)
    # perturb_x = np.loadtxt(args.perturb_x, dtype=np.float32)[args.index]
    # x0 = np.zeros_like(perturb_x)
    # betas = make_beta_schedule('linear',1000,1e-4,0.02)
    # alphas = 1 - betas
    # gammas = np.cumprod(alphas, axis=0)  # (0 - 1)
    # steps = np.linspace(0, 1000, args.steps,endpoint=False,dtype=np.int32)
    # for xi, t in tqdm(enumerate(steps),total=len(steps)):
    #     xt = x0 * np.sqrt(1-gammas[t]) + perturb_x * np.sqrt(gammas[t])
    #     extran = toMat(xt[:3],xt[3:]) @ gt_x
    #     proj_pcd, _, depth = npproj(pcd, extran, intran, img_hw, return_depth=True)
    #     viz_proj_pcd(proj_pcd, depth, image, str(res_dir.joinpath("%06d.png"%xi)))