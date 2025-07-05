import numpy as np
from matplotlib import pyplot as plt
import argparse
from PIL import Image
from models.tools.cmsc import toMat, npproj
from pathlib import Path
import os
import open3d as o3d
import shutil
from tqdm import tqdm

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",type=str,default="data/kitti/sequences/13/image_2")
    parser.add_argument("--lidar_dir",type=str,default="data/kitti/sequences/13/velodyne")
    parser.add_argument("--gt_mat",type=str,default="cache/kitti_gt/13_gt.txt")
    parser.add_argument("--x0_dir",type=str,default="experiments/kitti/nlsd/lccnet/results/nlsd_10_2025-02-02-10-24-47/seq_13")
    parser.add_argument("--index",type=int,default=223)
    parser.add_argument("--intran",type=float,nargs=3, default=[7.188560000000e+02, 6.071928000000e+02, 1.852157000000e+02], help='f, cx, cy')
    parser.add_argument("--res_dir",type=str,default="fig/nlsd")
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
    if res_dir.exists():
        shutil.rmtree(str(res_dir))
    res_dir.mkdir(parents=True)
    img_files = sorted(os.listdir(args.image_dir))
    lidar_files = sorted(os.listdir(args.lidar_dir))
    image = np.array(Image.open(os.path.join(args.image_dir, img_files[args.index])).convert('RGB'))
    x0_files = sorted(os.listdir(args.x0_dir))
    img_hw = image.shape[:2]
    x0 = np.loadtxt(os.path.join(args.x0_dir, x0_files[args.index]))
    assert np.ndim(x0) == 2
    pcd = loadpcd(os.path.join(args.lidar_dir, lidar_files[args.index]))
    f, cx, cy = args.intran
    intran = np.array([[f, 0, cx],
                      [0,f,cy],
                      [0,0,1]])
    gt_mat = np.loadtxt(args.gt_mat, dtype=np.float32)
    proj_pcd, _, depth = npproj(pcd, gt_mat, intran, img_hw, return_depth=True)
    viz_proj_pcd(proj_pcd, depth, image, str(res_dir.joinpath("gt.png")))
    for t, xt in tqdm(enumerate(x0), total=len(x0)):
        extran = toMat(xt[:3],xt[3:])
        proj_pcd, _, depth = npproj(pcd, extran, intran, img_hw, return_depth=True)
        viz_proj_pcd(proj_pcd, depth, image, str(res_dir.joinpath("%06d.png"%t)))