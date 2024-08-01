import os
import json
import torch
from torch.utils.data.dataset import Dataset 
from torchvision.transforms import transforms as Tf
import numpy as np
import pykitti
import open3d as o3d
from util import transform, se3
from PIL import Image
from torch import Generator, randperm
from torch.utils.data import Dataset, Subset
from typing import Iterable, Dict, Union, Optional, Tuple, Sequence

def subset_split(dataset:Dataset, lengths:Sequence[int], seed:Optional[int]=None):
	"""
	split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
	"""
	if seed is not None:
		generator = Generator().manual_seed(seed)
	else:
		generator = None
	indices = randperm(sum(lengths), generator=generator).tolist()
	Subsets = []
	for offset, length in zip(np.add.accumulate(lengths), lengths):
		if length == 0:
			Subsets.append(None)
		else:
			Subsets.append(Subset(dataset, indices[offset - length : offset]))
	return Subsets

def check_length(root:str,save_name='data_len.json'):
    seq_dir = os.path.join(root,'sequences')
    seq_list = os.listdir(seq_dir)
    seq_list.sort()
    dict_len = dict()
    for seq in seq_list:
        len_velo = len(os.listdir(os.path.join(seq_dir,seq,'velodyne')))
        dict_len[seq]=len_velo
    with open(os.path.join(root,save_name),'w')as f:
        json.dump(dict_len,f)
        
class KITTIFilter:
    def __init__(self, voxel_size=0.3, concat:str='none'):
        """KITTIFilter

        Args:
            voxel_size (float, optional): voxel size for downsampling. Defaults to 0.3.
            concat (str, optional): concat operation for normal estimation, 'none','xyz' or 'zero-mean'. Defaults to 'none'.
        """
        self.voxel_size = voxel_size
        self.concat = concat
        
    def __call__(self, x:np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x)
        # _, ind = pcd.remove_radius_outlier(nb_points=self.n_neighbor, radius=self.voxel_size)
        # pcd.select_by_index(ind)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        pcd_xyz = np.array(pcd.points,dtype=np.float32)
        if self.concat == 'none':
            return pcd_xyz
        else:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*3, max_nn=30))
            pcd.normalize_normals()
            pcd_norm = np.array(pcd.normals,dtype=np.float32)
            if self.concat == 'xyz':
                return np.hstack([pcd_xyz,pcd_norm])  # (N,3), (N,3) -> (N,6)
            elif self.concat == 'zero-mean':  # 'zero-mean'
                center = np.mean(pcd_xyz,axis=0,keepdims=True)  # (3,)
                pcd_zero = pcd_xyz - center
                pcd_norm *= np.where(np.sum(pcd_zero*pcd_norm,axis=1,keepdims=True)<0,-1,1)
                return np.hstack([pcd_zero,pcd_norm]) # (N,3),(N,3) -> (N,6)
            else:
                raise RuntimeError('Unknown concat mode: %s'%self.concat)

class Resampler:
    """ [N, D] -> [M, D]\n
    used for training
    """
    def __init__(self, num):
        self.num = num

    def __call__(self, x: np.ndarray):
        num_points = x.shape[0]
        idx = np.random.permutation(num_points)
        if self.num < 0:
            return x[idx]
        elif self.num <= num_points:
            idx = idx[:self.num] # (self.num,dim)
            return x[idx]
        else:
            idx = np.hstack([idx,np.random.choice(num_points,self.num-num_points,replace=True)]) # (self.num,dim)
            return x[idx]

class MaxResampler:
    """ [N, D] -> [M, D] (M<=max_num)\n
    used for testing
    """
    def __init__(self,num,seed=8080):
        self.num = num
        np.random.seed(seed)  # fix randomly sampling in test pipline
    def __call__(self, x:np.ndarray):
        num_points = x.shape[0]
        x_ = np.random.permutation(x)
        if num_points <= self.num:
            return x_  # permutation
        else:
            return x_[:self.num]

class ToTensor:
    def __init__(self,type=torch.float):
        self.tensor_type = type
    
    def __call__(self, x: np.ndarray):
        return torch.from_numpy(x).type(self.tensor_type)



class BaseKITTIDataset(Dataset):
    def __init__(self,basedir:str, batch_size:int,
                 seqs=['09','10'], cam_id:int=2,
                 meta_json='data_len.json', skip_frame=1,
                 voxel_size=0.15, pcd_sample_num=8192,
                 resize_size:Optional[Tuple[int,int]]=None, extend_ratio=(2.5,2.5),
                 ):
        if not os.path.exists(os.path.join(basedir,meta_json)):
            check_length(basedir,meta_json)
        with open(os.path.join(basedir,meta_json),'r')as f:
            dict_len = json.load(f)
        frame_list = []
        for seq in seqs:
            frame = list(range(0,dict_len[seq],skip_frame))
            cut_index = len(frame) % batch_size
            if cut_index > 0:
                frame = frame[:-cut_index]  # to be flexible to pykitti
            frame_list.append(frame)
        self.kitti_datalist = [pykitti.odometry(basedir,seq,frames=frame) for seq,frame in zip(seqs,frame_list)]  
        # concat images from different seq into one batch will cause error
        self.cam_id = cam_id
        self.resize_size = resize_size
        for seq,obj in zip(seqs,self.kitti_datalist):
            self.check(obj,cam_id,seq)
        self.sep = [len(data) for data in self.kitti_datalist]
        self.sumsep = np.cumsum(self.sep)
        self.resample_tran = Resampler(pcd_sample_num)
        self.tensor_tran = ToTensor()
        if self.resize_size is not None:
            self.img_tran = Tf.Compose([
                Tf.ToTensor(),
                Tf.Resize(self.resize_size)])
        else:
            self.img_tran = Tf.ToTensor()
        self.pcd_tran = KITTIFilter(voxel_size, 'none')
        self.extend_ratio = extend_ratio
        
    def __len__(self):
        return self.sumsep[-1]
    @staticmethod
    def check(odom_obj:pykitti.odometry,cam_id:int,seq:str)->bool:
        calib = odom_obj.calib
        cam_files_length = len(getattr(odom_obj,'cam%d_files'%cam_id))
        velo_files_lenght = len(odom_obj.velo_files)
        head_msg = '[Seq %s]:'%seq
        assert cam_files_length>0, head_msg+'None of camera %d files'%cam_id
        assert cam_files_length==velo_files_lenght, head_msg+"number of cam %d (%d) and velo files (%d) doesn't equal!"%(cam_id,cam_files_length,velo_files_lenght)
        assert hasattr(calib,'T_cam0_velo'), head_msg+"Crucial calib attribute 'T_cam0_velo' doesn't exist!"
        
    
    def __getitem__(self, index):
        group_id = np.digitize(index,self.sumsep,right=False)
        data = self.kitti_datalist[group_id]
        T_cam2velo = getattr(data.calib,'T_cam%d_velo'%self.cam_id)     
        if group_id > 0:
            sub_index = index - self.sumsep[group_id-1]
        else:
            sub_index = index
        raw_img:Image.Image = getattr(data,'get_cam%d'%self.cam_id)(sub_index)  # PIL Image
        H,W = raw_img.height, raw_img.width
        K_cam:np.ndarray = getattr(data.calib,'K_cam%d'%self.cam_id)  
        if self.resize_size is not None:
            RH, RW = self.resize_size
            K_cam = np.diag([RH / H, RW / W, 1.0]) @ K_cam
        else:
            RH = H
            RW = W
        REVH,REVW = self.extend_ratio[0]*RH,self.extend_ratio[1] * RW
        K_cam_extend = K_cam.copy()  # K_cam_extend for dilated projection
        K_cam_extend[0,-1] *= self.extend_ratio[0]
        K_cam_extend[1,-1] *= self.extend_ratio[1]
        raw_img = raw_img.resize([RW,RH],Image.BILINEAR)
        _img = self.img_tran(raw_img)  # raw img input (3,H,W)
        pcd:np.ndarray = data.get_velo(sub_index)
        pcd[:,3] = 1.0  # (N,4)
        calibed_pcd = T_cam2velo @ pcd.T  # [4,4] @ [4,N] -> [4,N]
        _calibed_pcd = self.pcd_tran(calibed_pcd[:3,:].T).T  # calibrated pcd input (3,N)
        *_,rev = transform.binary_projection((REVH,REVW), K_cam_extend,_calibed_pcd)
        _calibed_pcd = _calibed_pcd[:,rev]  
        _calibed_pcd = self.resample_tran(_calibed_pcd.T).T # (3,n)
        _calibed_pcd = self.tensor_tran(_calibed_pcd)
        T_cam2velo = self.tensor_tran(T_cam2velo)
        camera_info = {
            "f": K_cam[0,0].item(),
            "cx": K_cam[0,2].item(),
            "cy": K_cam[1,2].item(),
            "sensor_h": RH,
            "sensor_w": RW,
            "projection_mode": "perspective"
        }
        return dict(img=_img,pcd=_calibed_pcd, camera_info=camera_info, ExTran=T_cam2velo)
    
    @staticmethod
    def collate_fn(zipped_x:Iterable[Dict[str, Union[torch.Tensor, Dict]]]):
        batch = dict()
        batch['img'] = torch.stack([x['img'] for x in zipped_x])
        batch['pcd'] = torch.stack([x['pcd'] for x in zipped_x])
        batch['Extran'] = torch.stack([x['Extran'] for x in zipped_x])
        batch['camera_info'] = zipped_x[0]['camera_info']
        return batch
        
class PertubKITTIDataset(Dataset):
    def __init__(self,dataset:BaseKITTIDataset,
                 max_deg:float,
                 max_tran:float,
                 mag_randomly=True,
                 file=None):
        self.dataset = dataset
        self.file = file
        if self.file is not None:
            self.perturb = torch.from_numpy(np.loadtxt(self.file,dtype=np.float32,delimiter=','))[None,...]  # (1,N,6)
        else:
            self.transform = transform.UniformTransformSE3(max_deg, max_tran, mag_randomly)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        data = self.dataset[index]
        calibed_pcd = data['pcd']  # (3,N)
        if self.file is None:  # randomly generate igt
            _uncalib_pcd = self.transform(calibed_pcd[None,:,:]).squeeze(0)  # (3,N)
            gt = self.transform.igt.squeeze(0)  # (4,4)
        else:
            igt = se3.exp(self.perturb[:,index,:])  # (1,6) -> (1,4,4)
            _uncalib_pcd = se3.transform(igt,calibed_pcd[None,...]).squeeze(0)  # (3,N)
            gt = transform.inv_pose(igt).squeeze(0)
        new_data = dict(img=data['img'],uncalib_pcd=_uncalib_pcd, gt=gt, camera_info=data['camera_info'])
        return new_data
    
    @staticmethod
    def collate_fn(zipped_x:Iterable[Dict[str, Union[torch.Tensor, Dict]]]):
        batch = dict()
        batch['img'] = torch.stack([x['img'] for x in zipped_x])
        batch['uncalib_pcd'] = torch.stack([x['uncalib_pcd'] for x in zipped_x])
        batch['gt'] = torch.stack([x['gt'] for x in zipped_x])
        batch['camera_info'] = zipped_x[0]['camera_info']
        return batch

        
        
if __name__ == "__main__":
    base_dataset = BaseKITTIDataset('data/kitti', 1, seqs=['00','01'], skip_frame=3)
    dataset = PertubKITTIDataset(base_dataset, 30, 0.3, True)
    data = dataset[0]
    for key,value in data.items():
        if hasattr(value, 'shape'):
            shape = value.shape
        else:
            shape = value
        print('{key}: {shape}'.format(key=key, shape=shape))
    
        

        