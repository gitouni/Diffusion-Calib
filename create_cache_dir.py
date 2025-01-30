from pathlib import Path
import shutil
from dataset import NuSceneDataset
import numpy as np
from tqdm import tqdm
cache_dir = Path("cache/nuscenes_gt")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
cache_dir.mkdir(parents=True)
dataset = NuSceneDataset('v1.0-test','data/nuscenes/')
for dataset, name in tqdm(dataset.split_dataset(), total=len(dataset.scene_name_list)):
    data = dataset[0]
    gt_matrix:np.ndarray = data['extran'].cpu().detach().numpy()
    np.savetxt(str(cache_dir.joinpath(name+'.txt')), gt_matrix, fmt='%06f')