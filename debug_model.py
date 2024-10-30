from models.denoiser import LCCRAFT
import torch

model = LCCRAFT(dict(),12).cuda()
img = torch.rand(2,3,256,512).cuda()
pcd = torch.rand(2,3,2000).cuda()
Tcl = torch.eye(4)[None].repeat(2,1,1).cuda()
camera_info = {'fx': 405.2535049748973, 'fy': 360.22533775546424, 'cx': 261.2054463183355, 'cy': 139.80645426999396, 'sensor_h': 256, 'sensor_w': 512, 'projection_mode': 'perspective'}
x0_list = model.forward(img, pcd, Tcl, camera_info)
for x0 in x0_list:
    print(x0.shape)