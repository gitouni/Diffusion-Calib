import pykitti
import os
import re
import numpy as np

save_dir = 'cache/kitti_gt'
kitti_dir = 'data/kitti'
dirnames = ['13','14','15','20','21']
dirnames.sort()
cam_id = 2

for subdir in dirnames:
    dataset = pykitti.odometry(kitti_dir, sequence=subdir)
    extran =  getattr(dataset.calib,'T_cam%d_velo'%cam_id)
    np.savetxt(os.path.join(save_dir, '{}_gt.txt'.format(subdir)), extran, fmt="%0.6f")