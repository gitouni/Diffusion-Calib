from dataset import NuSceneDataset
from models.tools.cmsc import npproj
import numpy as np
from matplotlib import pyplot as plt
# from nuscenes.nuscenes import NuScenes
if __name__ == "__main__":
    dataset = NuSceneDataset(resize_size=(256, 512), pcd_sample_num=40000, extend_ratio=[2.5, 2.5])
    img, pcd, extran, intran = dataset.group_sub_item(0,0)
    H, W = img.height, img.width
    uv, rev, depth = npproj(pcd, extran, intran, (img.height, img.width), return_depth=True)
    plt.figure(figsize=(8,5),dpi=100,tight_layout=True)
    plt.axis([0,W,H,0])
    plt.imshow(img)
    plt.scatter([uv[:,0]],[uv[:,1]],c=[depth],cmap='rainbow_r',alpha=0.8,s=5)
    plt.show()
    # nusc = NuScenes(version='v1.0-mini', dataroot='data/nuScenes', verbose=False)
    # nusc.render_pointcloud_in_image('ca9a282c9e77460f8360f564131a8af5',pointsensor_channel='LIDAR_TOP')