import numpy as np
from matplotlib import pyplot as plt
import torch
feat = torch.load('debug_proj_feat.pt').cpu().detach().numpy()
feat_vis = np.mean(feat[0], axis=0)
plt.imshow(feat_vis)
plt.axis('off')
plt.savefig("fig/method/proj_feat.png", bbox_inches='tight')