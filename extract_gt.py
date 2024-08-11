import pykitti
import os
import re
import numpy as np

colmap_dir = 'data/colmap'
kitti_dir = 'data/kitti'
dirnames = [dirname for dirname in os.listdir(colmap_dir) if re.search('\d+', dirname)]
dirnames.sort()
cam_id = 2
print("valid seq names:{}".format(dirnames))
for subdir in dirnames:
    dataset = pykitti.odometry(kitti_dir, sequence=subdir)
    extran =  getattr(dataset.calib,'T_cam%d_velo'%cam_id)
    np.savetxt(os.path.join(colmap_dir, subdir, 'gt.txt'), extran, fmt="%0.6f")