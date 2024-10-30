import numpy as np
from matplotlib import pyplot as plt
import torch
raw_feat = torch.load('debug/tensor/raw_feat.pt').cpu().detach().numpy()
rotperturb_feat = torch.load('debug/tensor/rotperturb_feat.pt').cpu().detach().numpy()
tslperturb_feat = torch.load('debug/tensor/tslperturb_feat.pt').cpu().detach().numpy()
# featmap sensistivity to rotation perturbation
delta_F = rotperturb_feat - raw_feat
delta_Fp = delta_F[:,512:768,...]
delta_Fd = delta_F[:,-256:,...]
tags = ['roll','pitch','yaw']
for i, tag in enumerate(tags):
    feat_dFp = np.mean(delta_Fp[i], axis=0)
    plt.imshow(feat_dFp)
    plt.axis('off')
    plt.savefig("fig/tensor/Fp_{}_diff.png".format(tag), bbox_inches='tight')
    feat_dFd = np.mean(delta_Fd[i], axis=0)
    plt.imshow(feat_dFd)
    plt.axis('off')
    plt.savefig("fig/tensor/Fd_{}_diff.png".format(tag), bbox_inches='tight')
# featmap sensistivity to translation perturbation
delta_F = tslperturb_feat - raw_feat
delta_Fp = delta_F[:,512:768,...]
delta_Fd = delta_F[:,-256:,...]
tags = ['tx','ty','tz']
for i, tag in enumerate(tags):
    feat_dFp = np.mean(delta_Fp[i], axis=0)
    plt.imshow(feat_dFp)
    plt.axis('off')
    plt.savefig("fig/tensor/Fp_{}_diff.png".format(tag), bbox_inches='tight')
    feat_dFd = np.mean(delta_Fd[i], axis=0)
    plt.imshow(feat_dFd)
    plt.axis('off')
    plt.savefig("fig/tensor/Fd_{}_diff.png".format(tag), bbox_inches='tight')