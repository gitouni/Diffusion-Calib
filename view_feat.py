import numpy as np
from matplotlib import pyplot as plt
feat1 = np.mean(np.load('debug1_1.npy')[0], axis=0)
feat2 = np.mean(np.load('debug1_2.npy')[0], axis=0)
plt.imshow(feat1-feat2)
plt.show()