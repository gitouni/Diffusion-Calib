import os
import json
import torch
from PIL import Image
from torchvision.transforms import transforms as Tf
import numpy as np
import pykitti
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.map_mask import MapMask
# from nuscenes.utils.color_map import get_colormap
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d
from models.util import transform, se3
from PIL import Image
from torch import Generator, randperm
from torch.utils.data import Dataset, Subset, BatchSampler
from typing import Iterable, List, Dict, Union, Optional, Tuple, Sequence, Literal
# from models.colmap.io import read_model, CAMERA_TYPE
from models.tools.cmsc import nptran
from models.util.transform import inv_pose_np
# from RANSAC.base import RotEstimator,TslEstimator,RotRANSAC, TslRANSAC
import re
import time

IMAGENET_MEAN=[0.485, 0.456, 0.406]
IMAGENET_STD=[0.229, 0.224, 0.225]



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
    def __init__(self, voxel_size:Optional[float]=None, min_dist:float=0.15, skip_point:int=1):
        """KITTIFilter

        Args:
            voxel_size (float, optional): voxel size for downsampling. Defaults to 0.3.
            concat (str, optional): concat operation for normal estimation, 'none','xyz' or 'zero-mean'. Defaults to 'none'.
        """
        self.voxel_size = voxel_size
        self.min_dist = min_dist
        self.skip_point = skip_point
        
    def __call__(self, x:np.ndarray):
        if self.skip_point > 1:
            x = x[::self.skip_point,:]
        rev_x = np.linalg.norm(x, axis=1) > self.min_dist
        
        # _, ind = pcd.remove_radius_outlier(nb_points=self.n_neighbor, radius=self.voxel_size)
        # pcd.select_by_index(ind)
        if self.voxel_size is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(x[rev_x,:])
            pcd = pcd.voxel_down_sample(self.voxel_size)
            pcd_xyz = np.array(pcd.points,dtype=np.float32)
        else:
            pcd_xyz = x[rev_x,:]
        return pcd_xyz

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
    def __init__(self,num):
        self.num = num
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


class KITTIBatchSampler(BatchSampler):
    def __init__(self, num_sequences:int, len_of_sequences:Sequence[int], dataset_len:int, num_samples:int=4):
        # Batch sampler with a dynamic number of sequences
        # max_images >= number_of_sequences * images_per_sequence
        assert num_sequences == len(len_of_sequences)
        self.num_samples = num_samples
        self.num_sequences = num_sequences
        self.len_of_sequences = len_of_sequences
        self.dataset_len = dataset_len

    def __iter__(self):
        for _ in range(self.dataset_len):
            # number per sequence
            seq_indices = np.random.choice(self.num_sequences, self.num_samples, replace=True)
            batches = [(seq_idx, np.random.choice(self.len_of_sequences[seq_idx], 1).item()) for seq_idx in seq_indices]
            yield batches

    def __len__(self):
        return self.dataset_len
    


# class PerturbDataset(Dataset):
#     def __init__(self,dataset:Dataset,
#                  max_deg:float,
#                  max_tran:float,
#                  mag_randomly=True,
#                  file=None):
#         self.dataset = dataset
#         self.file = file
#         if self.file is not None:
#             if os.path.isfile(self.file):
#                 self.perturb = torch.from_numpy(np.loadtxt(self.file, dtype=np.float32))[None,...]  # (1,N,6)
#             else:
#                 random_transform = transform.UniformTransformSE3(max_deg, max_tran, mag_randomly)
#                 perturb = random_transform.generate_transform(len(dataset))
#                 np.savetxt(self.file, perturb.cpu().detach().numpy(), fmt='%0.6f')
#                 self.perturb = perturb.unsqueeze(0)  # (1,N,6)
#         else:
#             self.transform = transform.UniformTransformSE3(max_deg, max_tran, mag_randomly)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index:int):
#         data = self.dataset[index]
#         if self.file is None:  # randomly generate igt
#             igt_x = self.transform.generate_transform(1)
#             igt = se3.exp(igt_x).squeeze(0)
#             gt = transform.inv_pose(igt)
#         else:
#             igt = se3.exp(self.perturb[:,index,:]).squeeze(0)  # (1,6) -> (1,4,4)
#             gt = transform.inv_pose(igt).squeeze(0)
#         extran = igt @ data['extran']  # add noise to the ground-truth extran
#         data.update(dict(gt=gt, extran=extran))  # extran here denotes the perturbed gt extran
#         return data



class BaseKITTIDataset(Dataset):
    def __init__(self,basedir:str,
                 seqs:List[str]=['09','10'], cam_id:int=2,
                 meta_json:str='data_len.json', skip_frame:int=1, skip_point:int=1,
                 voxel_size:Optional[float]=None, min_dist=0.1, pcd_sample_num=8192,
                 resize_size:Optional[Tuple[int,int]]=None, extend_ratio=(2.5,2.5),
                 ):
        if not os.path.exists(os.path.join(basedir,meta_json)):
            check_length(basedir,meta_json)
        with open(os.path.join(basedir,meta_json),'r')as f:
            dict_len = json.load(f)
        frame_list = []
        for seq in seqs:
            frame = list(range(0,dict_len[seq],skip_frame))
            # cut_index = len(frame) % batch_size
            # if cut_index > 0:
            #     frame = frame[:-cut_index]  # to be flexible to pykitti
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
        self.tensor_tran = lambda x:torch.from_numpy(x).to(torch.float32)
        self.img_tran = Tf.Compose([Tf.ToTensor(),
                                    Tf.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.pcd_tran = KITTIFilter(voxel_size, min_dist, skip_point)
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
        
    def __getitem__(self, index:Union[int, Tuple[int,int]]):
        if isinstance(index, Tuple):
            return self.group_sub_item(index)
        group_id = np.digitize(index,self.sumsep,right=False)
        if group_id > 0:
            sub_index = index - self.sumsep[group_id-1]
        else:
            sub_index = index
        return self.group_sub_item((group_id, sub_index))
        
    def group_sub_item(self, tuple_index:Tuple[int,int]):
        group_idx, sub_index = tuple_index
        data = self.kitti_datalist[group_idx]
        T_cam2velo = getattr(data.calib,'T_cam%d_velo'%self.cam_id)  
        raw_img:Image.Image = getattr(data,'get_cam%d'%self.cam_id)(sub_index)  # PIL Image
        H,W = raw_img.height, raw_img.width
        K_cam:np.ndarray = getattr(data.calib,'K_cam%d'%self.cam_id)  
        if self.resize_size is not None:
            RH, RW = self.resize_size
            K_cam = np.diag([RW / W, RH / H, 1.0]) @ K_cam
        else:
            RH = H
            RW = W
       
        raw_img = raw_img.resize([RW,RH],Image.Resampling.BILINEAR)
        _img = self.img_tran(raw_img)  # raw img input (3,H,W)
        pcd:np.ndarray = data.get_velo(sub_index)[:,:3]
        pcd = self.pcd_tran(pcd)
        if self.extend_ratio is not None:
            calibed_pcd = nptran(pcd, T_cam2velo).T
            REVH,REVW = self.extend_ratio[0]*RH,self.extend_ratio[1] * RW
            K_cam_extend = K_cam.copy()  # K_cam_extend for dilated projection
            K_cam_extend[0,-1] *= self.extend_ratio[0]
            K_cam_extend[1,-1] *= self.extend_ratio[1]
            *_,rev = transform.binary_projection((REVH,REVW), K_cam_extend, calibed_pcd)
            pcd = pcd[rev,:]
        pcd = self.resample_tran(pcd) # (n,3)
        _pcd = self.tensor_tran(pcd.T)
        T_cam2velo = self.tensor_tran(T_cam2velo)
        camera_info = {
            "fx": K_cam[0,0].item(),
            "fy": K_cam[1,1].item(),
            "cx": K_cam[0,2].item(),
            "cy": K_cam[1,2].item(),
            "sensor_h": RH,
            "sensor_w": RW,
            "projection_mode": "perspective"
        }
        return dict(img=_img,pcd=_pcd, camera_info=camera_info, extran=T_cam2velo, group_idx=group_idx, sub_index=sub_index)
    
    @staticmethod
    def collate_fn(zipped_x:Iterable[Dict[str, Union[torch.Tensor, Dict]]]):
        batch = dict()
        batch['img'] = torch.stack([x['img'] for x in zipped_x])
        batch['pcd'] = torch.stack([x['pcd'] for x in zipped_x])
        batch['extran'] = torch.stack([x['extran'] for x in zipped_x])
        batch['group_idx'] = [x['group_idx'] for x in zipped_x]
        batch['sub_index'] = [x['sub_index'] for x in zipped_x]
        batch['camera_info'] = zipped_x[0]['camera_info']
        batch['camera_info']['fx'] = torch.tensor([x['camera_info']['fx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['fy'] = torch.tensor([x['camera_info']['fy'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cx'] = torch.tensor([x['camera_info']['cx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cy'] = torch.tensor([x['camera_info']['cy'] for x in zipped_x], dtype=torch.float32)
        return batch
        
class PerturbDataset(Dataset):
    def __init__(self,dataset:BaseKITTIDataset,
                 max_deg:float,
                 max_tran:float,
                 mag_randomly=True,
                 file=None):
        self.dataset = dataset
        self.file = file
        if self.file is not None:
            if os.path.isfile(self.file):
                self.perturb = torch.from_numpy(np.loadtxt(self.file, dtype=np.float32))[None,...]  # (1,N,6)
            else:
                random_transform = transform.UniformTransformSE3(max_deg, max_tran, mag_randomly)
                perturb = random_transform.generate_transform(len(dataset))
                np.savetxt(self.file, perturb.cpu().detach().numpy(), fmt='%0.6f')
                self.perturb = perturb.unsqueeze(0)  # (1,N,6)
        else:
            self.transform = transform.UniformTransformSE3(max_deg, max_tran, mag_randomly)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index:Union[int, Tuple[int,int]]):
        data = self.dataset[index]
        if isinstance(index, Tuple):
            group_idx, sub_index = index
            total_index = self.dataset.sumsep[group_idx] + sub_index
        else:
            total_index = index
        extran = data['extran']  # (4,4)
        if self.file is None:  # randomly generate igt
            igt_x = self.transform.generate_transform(1)
            igt = se3.exp(igt_x).squeeze(0)
            gt = transform.inv_pose(igt)
        else:
            igt = se3.exp(self.perturb[:,total_index,:]).squeeze(0)  # (1,6) -> (1,4,4)
            gt = transform.inv_pose(igt).squeeze(0)
        extran = igt @ extran
        new_data = dict(img=data['img'],pcd=data['pcd'], gt=gt, extran=extran, camera_info=data['camera_info'],
                        group_idx=data['group_idx'], sub_index=data['sub_index'])
        return new_data
    
    @staticmethod
    def collate_fn(zipped_x:Iterable[Dict[str, Union[torch.Tensor, Dict]]]):
        batch = dict()
        batch['img'] = torch.stack([x['img'] for x in zipped_x])
        batch['pcd'] = torch.stack([x['pcd'] for x in zipped_x])
        batch['group_idx'] = [x['group_idx'] for x in zipped_x]
        batch['sub_index'] = [x['sub_index'] for x in zipped_x]
        batch['extran'] = torch.stack([x['extran'] for x in zipped_x])
        batch['gt'] = torch.stack([x['gt'] for x in zipped_x])
        batch['camera_info'] = zipped_x[0]['camera_info']
        batch['camera_info']['fx'] = torch.tensor([x['camera_info']['fx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['fy'] = torch.tensor([x['camera_info']['fy'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cx'] = torch.tensor([x['camera_info']['cx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cy'] = torch.tensor([x['camera_info']['cy'] for x in zipped_x], dtype=torch.float32)
        return batch
   
class LightNuscenes(NuScenes):
    def __init__(self, version = 'v1.0-mini', dataroot = '/data/sets/nuscenes', verbose = True, map_resolution = 0.1):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0", ...).
        :param dataroot: Path to the tables and data.
        :param verbose: Whether to print status messages during load.
        :param map_resolution: Resolution of maps (meters).
        """
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.table_names = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                            'log', 'scene', 'sample', 'sample_data','map']

        assert os.path.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)

        start_time = time.time()
        if verbose:
            print("======\nLoading NuScenes tables for version {}...".format(self.version))

        # Explicitly assign tables to help the IDE determine valid class members.
        for name in self.table_names:
            setattr(self, name, self.__load_table__(name))  # remove unnecessary keys: ego_pose, sample_annotation, map
        # self.category = self.__load_table__('category')
        # self.attribute = self.__load_table__('attribute')
        # self.visibility = self.__load_table__('visibility')
        # self.instance = self.__load_table__('instance')
        # self.sensor = self.__load_table__('sensor')
        # self.calibrated_sensor = self.__load_table__('calibrated_sensor')
        # self.ego_pose = self.__load_table__('ego_pose')
        # self.log = self.__load_table__('log')
        # self.scene = self.__load_table__('scene')
        # self.sample = self.__load_table__('sample')
        # self.sample_data = self.__load_table__('sample_data')
        # self.sample_annotation = self.__load_table__('sample_annotation')
        # self.map = self.__load_table__('map')

        # Initialize the colormap which maps from class names to RGB values.
        # self.colormap = get_colormap()

        # lidar_tasks = [t for t in ['lidarseg', 'panoptic'] if os.path.exists(os.path.join(self.table_root, t + '.json'))]
        # if len(lidar_tasks) > 0:
        #     self.lidarseg_idx2name_mapping = dict()
        #     self.lidarseg_name2idx_mapping = dict()
        #     self.load_lidarseg_cat_name_mapping()
        # for i, lidar_task in enumerate(lidar_tasks):
        #     if self.verbose:
        #         print(f'Loading nuScenes-{lidar_task}...')
        #     if lidar_task == 'lidarseg':
        #         self.lidarseg = self.__load_table__(lidar_task)
        #     else:
        #         self.panoptic = self.__load_table__(lidar_task)

        #     setattr(self, lidar_task, self.__load_table__(lidar_task))
        #     label_files = os.listdir(os.path.join(self.dataroot, lidar_task, self.version))
        #     num_label_files = len([name for name in label_files if (name.endswith('.bin') or name.endswith('.npz'))])
        #     num_lidarseg_recs = len(getattr(self, lidar_task))
        #     assert num_lidarseg_recs == num_label_files, \
        #         f'Error: there are {num_label_files} label files but {num_lidarseg_recs} {lidar_task} records.'
        #     self.table_names.append(lidar_task)
        #     # Sort the colormap to ensure that it is ordered according to the indices in self.category.
        #     self.colormap = dict({c['name']: self.colormap[c['name']]
        #                           for c in sorted(self.category, key=lambda k: k['index'])})

        # # If available, also load the image_annotations table created by export_2d_annotations_as_json().
        # if os.path.exists(os.path.join(self.table_root, 'image_annotations.json')):
        #     self.image_annotations = self.__load_table__('image_annotations')

        # Initialize map mask for each map record.
        for map_record in self.map:
            map_record['mask'] = MapMask(os.path.join(self.dataroot, map_record['filename']), resolution=map_resolution)

        if verbose:
            for table in self.table_names:
                print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

        # Initialize NuScenesExplorer class.
        self.explorer = NuScenesExplorer(self)

    def __make_reverse_index__(self, verbose: bool) -> None:
        """
        De-normalizes database to create reverse indices for common cases.
        :param verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member['token']] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        # for record in self.sample_annotation:
        #     inst = self.get('instance', record['instance_token'])
        #     record['category_name'] = self.get('category', inst['category_token'])['name']

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record['data'] = {}
            record['anns'] = []

        for record in self.sample_data:
            if record['is_key_frame']:
                sample_record = self.get('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']

        # for ann_record in self.sample_annotation:
        #     sample_record = self.get('sample', ann_record['sample_token'])
        #     sample_record['anns'].append(ann_record['token'])

        # Add reverse indices from log records to map records.
        if 'log_tokens' not in self.map[0].keys():
            raise Exception('Error: log_tokens not in map table. This code is not compatible with the teaser dataset.')
        log_to_map = dict()
        for map_record in self.map:
            for log_token in map_record['log_tokens']:
                log_to_map[log_token] = map_record['token']
        for log_record in self.log:
            log_record['map_token'] = log_to_map[log_record['token']]

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))

class NuSceneDataset(Dataset):
    def __init__(self, version='v1.0-trainval', dataroot='data/nuscenes/v1.0-full',
            scene_names:Optional[Union[str,List[str]]]=None, daylight:bool=True,
            cam_sensor_name:Literal['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']='CAM_FRONT',
            point_sensor_name:str='LIDAR_TOP', skip_point:int=1,
            voxel_size:Optional[float]=None, min_dist=0.15, pcd_sample_num=8192,
            resize_size:Optional[Tuple[int,int]]=None, extend_ratio:Optional[Tuple[float,float]]=None) -> None:
        self.nusc = LightNuscenes(version=version, dataroot=dataroot, verbose=True)
        self.cam_sensor_name = cam_sensor_name
        self.point_sensor_name = point_sensor_name
        if isinstance(scene_names, str):
            if os.path.exists(scene_names):
                scene_names = np.loadtxt(scene_names, dtype=str).tolist()
            else:
                scene_names = [scene_names]
        if scene_names is not None:
            self.select_scene = [scene for scene in self.nusc.scene if scene['name'] in scene_names]
        else:
            self.select_scene = [scene for scene in self.nusc.scene if (not daylight) or (re.search('night',scene['description'].lower()) is None)]
        self.scene_num_list = [scene['nbr_samples'] for scene in self.select_scene]
        self.sumsep = np.cumsum(self.scene_num_list)
        self.sample_tokens_by_scene = []
        for scene, nbr_sample in zip(self.select_scene, self.scene_num_list):
            sample_tokens = []
            first_sample_token = scene['first_sample_token']
            sample_tokens.append(first_sample_token)
            sample_token  = first_sample_token
            for _ in range(nbr_sample - 1):
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']
                sample_tokens.append(sample_token)
            self.sample_tokens_by_scene.append(sample_tokens)
        self.img_tran = Tf.Compose([
             Tf.ToTensor(),
             Tf.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.pcd_filter = KITTIFilter(voxel_size, min_dist, skip_point)
        self.tensor_tran = lambda x:torch.from_numpy(x).to(torch.float32)
        self.resample_tran = Resampler(pcd_sample_num)
        self.resize_size = resize_size
        self.extend_ratio = extend_ratio
        
    def __len__(self):
        return self.sumsep[-1]
    
    def get_seq_params(self) -> Tuple[int, List[int]]:
        """Get num_seqs, num_data_per_seq

        Returns:
            Tuple[int, List[int]]: num_seqs, num_data_per_seq
        """
        return len(self.scene_num_list), self.scene_num_list
    
    def __getitem__(self, index:Union[int, Tuple[int,int]]):
        if isinstance(index, Tuple):
            return self.group_sub_item(*index)
        group_id = np.digitize(index,self.sumsep,right=False).item()
        if group_id > 0:
            sub_idx = index - self.sumsep[group_id-1]
        else:
            sub_idx = index
        return self.group_sub_item(group_id, sub_idx)

    def group_sub_item(self, group_idx:int, sub_idx:int):
        token = self.sample_tokens_by_scene[group_idx][sub_idx]
        sample = self.nusc.get('sample', token)
        img, pcd, extran, intran = self.get_data(sample, self.cam_sensor_name, self.point_sensor_name)
        camera_info = {
            "fx": intran[0,0].item(),
            "fy": intran[1,1].item(),
            "cx": intran[0,2].item(),
            "cy": intran[1,2].item(),
            "sensor_h": img.height,
            "sensor_w": img.width,
            "projection_mode": "perspective"
        }
        _img = self.img_tran(img)
        _pcd = self.tensor_tran(pcd.T)  # (3,N)
        extran = self.tensor_tran(extran)
        return dict(img=_img,pcd=_pcd, camera_info=camera_info, extran=extran, group_idx=group_idx, sub_idx=sub_idx)
    
    def get_data(self, sample_record:Dict,
            camera_channel:Literal['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT'],
            pointsensor_channel:Literal['LIDAR_TOP']) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
        pointsensor_token = sample_record['data'][pointsensor_channel]
        camera_token = sample_record['data'][camera_channel]
        cam = self.nusc.get('sample_data', camera_token)
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        img_path = os.path.join(self.nusc.dataroot, cam['filename'])
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor['filename'])
        img = Image.open(img_path)
        pc = LidarPointCloud.from_file(pcl_path)
        pcd = np.copy(pc.points).transpose(1,0)[:,:3]
        lidar_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        cam_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pose_lidar = np.eye(4)
        pose_lidar[:3,:3] = Quaternion(lidar_record['rotation']).rotation_matrix
        pose_lidar[:3,3] = np.array(lidar_record['translation'])
        pose_rgb = np.eye(4)
        pose_rgb[:3,:3] = Quaternion(cam_record['rotation']).rotation_matrix
        pose_rgb[:3,3] = np.array(cam_record['translation'])
        extran = inv_pose_np(pose_rgb) @ pose_lidar
        intran =  np.array(cam_record['camera_intrinsic'])
        H, W = img.height, img.width
        if self.resize_size is not None:
            RH, RW = self.resize_size[0], self.resize_size[1]
        else:
            RH, RW = H, W
        kx, ky = RW / W, RH / H
        intran[0,:] *= kx
        intran[1,:] *= ky
        if self.extend_ratio is not None:
            REVH,REVW = self.extend_ratio[0]*RH,self.extend_ratio[1] * RW
            K_cam_extend = intran.copy()  # K_cam_extend for dilated projection
            K_cam_extend[0,-1] *= self.extend_ratio[0]
            K_cam_extend[1,-1] *= self.extend_ratio[1]
            calibed_pcd = nptran(pcd, extran)
            *_,rev = transform.binary_projection((REVH,REVW), K_cam_extend, calibed_pcd.T)  # input pcd is (3, N)
            pcd = pcd[rev,:]
        pcd = self.resample_tran(pcd) # (n,3)
        img = img.resize([RW,RH],Image.Resampling.BILINEAR)
        # _img = self.img_tran(img)  # raw img input (3,H,W)
        return img, pcd, extran, intran

    @staticmethod
    def collate_fn(zipped_x:Iterable[Dict[str, Union[torch.Tensor, Dict]]]):
        batch = dict()
        batch['img'] = torch.stack([x['img'] for x in zipped_x])
        batch['pcd'] = torch.stack([x['pcd'] for x in zipped_x])
        batch['extran'] = torch.stack([x['extran'] for x in zipped_x])
        batch['group_idx'] = [x['group_idx'] for x in zipped_x]
        batch['sub_idx'] = [x['sub_idx'] for x in zipped_x]
        batch['camera_info'] = zipped_x[0]['camera_info']
        batch['camera_info']['fx'] = torch.tensor([x['camera_info']['fx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['fy'] = torch.tensor([x['camera_info']['fy'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cx'] = torch.tensor([x['camera_info']['cx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cy'] = torch.tensor([x['camera_info']['cy'] for x in zipped_x], dtype=torch.float32)
        return batch

__classdict__ = {'kitti':BaseKITTIDataset, 'nuscenes':NuSceneDataset}

if __name__ == "__main__":
    base_dataset = BaseKITTIDataset('data/kitti', seqs=['00','01'], skip_frame=3)
    dataset = PerturbDataset(base_dataset, 30, 0.3, True)
    data = dataset[0]
    for key,value in data.items():
        if hasattr(value, 'shape'):
            shape = value.shape
        else:
            shape = value
        print('{key}: {shape}'.format(key=key, shape=shape))
    
        


# class CBADataset(Dataset):
#     def __init__(self, gt_Tcl:str, image_dir:str, lidar_dir:str, lidar_pose:str, model_dir:str,
#              pair_file:str, kpt_dir:str, match_dir:str, max_frame_corr:int,
#              filter_params:Optional[Dict[str,float]]=None, pcd_sample_num:int=-1
#              ):
#         self.gt_Tcl = np.loadtxt(gt_Tcl)
#         image_files = sorted(os.listdir(image_dir))
#         lidar_files = sorted(os.listdir(lidar_dir))
#         kpt_files = sorted(os.listdir(kpt_dir))
#         cameras, images, points3d = read_model(model_dir, '.bin')
#         camera_id = cameras.keys().__iter__().__next__()
#         camera_data = cameras[camera_id]
#         assert camera_data.model in CAMERA_TYPE.keys(), 'Unknown camera type:{}'.format(camera_data.model)
#         params_dict = {key: value for key, value in zip(CAMERA_TYPE[camera_data.model], camera_data.params)}
#         if 'f' in params_dict:
#             fx = fy = params_dict['f']
#         else:
#             fx = params_dict['fx']
#             fy = params_dict['fy']
#         cx = params_dict['cx']
#         cy = params_dict['cy']
#         assert len(images.keys()) == len(image_files) == len(lidar_files), "images ({}), image_files ({}), lidar_files ({})".format(len(images.keys()), len(image_files), len(lidar_files))
#         self.img_files = [os.path.join(image_dir, file) for file in image_files]
#         self.lidar_files = [os.path.join(lidar_dir, file) for file in lidar_files]
#         self.kpts = [np.load(os.path.join(kpt_dir, file))['keypoints'] for file in kpt_files]
#         matches = [np.load(os.path.join(match_dir, file))['match'] for file in sorted(os.listdir(match_dir))]
#         self.cam_poses = []
#         self.cam_mappoints = []
#         for i in range(1,len(image_files)+1):
#             img_data = images[i]
#             extrinsics = np.eye(4)
#             extrinsics[:3,:3] = img_data.qvec2rotmat()
#             extrinsics[:3,3] = img_data.tvec
#             self.cam_poses.append(extrinsics)
#             point3d_ids = img_data.point3D_ids
#             point3d_valid_mask = np.nonzero(point3d_ids != -1)[0]
#             point3d_ids = point3d_ids[point3d_valid_mask]
#             mapppoints = np.stack([points3d[idx].xyz for idx in point3d_ids], axis=0)
#             mapppoints = nptran(mapppoints, extrinsics)  # frame coordinate system
#             self.cam_mappoints.append(mapppoints)
#         pose_graph = o3d.io.read_pose_graph(lidar_pose)
#         lidar_pose = [inv_pose_np(node.pose) for node in pose_graph.nodes]
#         N = len(pose_graph.nodes)
#         lidar_edge = [src_pose @ inv_pose_np(tgt_pose) for src_pose, tgt_pose in zip(lidar_pose[:-1], lidar_pose[1:])]
#         camera_edge = [src_pose @ inv_pose_np(tgt_pose) for src_pose, tgt_pose in zip(self.cam_poses[:-1], self.cam_poses[1:])]
#         camera_rotedge, lidar_rotedge = map(lambda edge_list:[T[:3,:3] for T in edge_list],[camera_edge, lidar_edge])
#         ransac_state = dict(min_samples=3,
#                                 max_trials=5000,
#                                 stop_prob=0.99,
#                                 random_state=0)
#         ransac_rot_estimator = RotRANSAC(RotEstimator(),
#                                 **ransac_state)
#         alpha,beta = map(ransac_rot_estimator.toVecList,[camera_rotedge, lidar_rotedge])
#         best_rot, rot_inlier_mask = ransac_rot_estimator.fit(beta,alpha)
#         ransac_tsl_estimator = TslRANSAC(TslEstimator(best_rot),
#                                      **ransac_state)
#         inlier_camera_edge = [edge for i,edge in enumerate(camera_edge) if rot_inlier_mask[i]]
#         inlier_lidar_edge = [edge for i,edge in enumerate(lidar_edge) if rot_inlier_mask[i]]
#         camera_flatten = ransac_tsl_estimator.flatten(inlier_camera_edge)
#         lidar_flatten = ransac_tsl_estimator.flatten(inlier_lidar_edge)
#         _, best_scale, _ = ransac_tsl_estimator.fit(camera_flatten, lidar_flatten)
#         self.scale = best_scale
#         self.pair = json.load(open(pair_file, 'r'))
#         self.pair_dict = dict()
#         for i, pair in enumerate(self.pair):
#             src_idx, tgt_idx = pair
#             if tgt_idx - src_idx > max_frame_corr:
#                 continue
#             if src_idx not in self.pair_dict.keys():
#                 self.pair_dict[src_idx] = []
#             self.pair_dict[src_idx].append([tgt_idx, matches[i]])
#             if tgt_idx not in self.pair_dict.keys():
#                 self.pair_dict[tgt_idx] = []
#             self.pair_dict[tgt_idx].append([src_idx, matches[i][:,::-1]])  # swap source and target
#         self.img_tran = Tf.Compose([
#              Tf.ToTensor(),
#              Tf.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
#         self.image_shape = [camera_data.height, camera_data.width]
#         self.pcd_tran = KITTIFilter(**filter_params) if filter_params else lambda x:x
#         self.resample_tran = Resampler(pcd_sample_num)
#         self.camera_info = {
#             "fx": fx,
#             "fy": fy,
#             "cx": cx,
#             "cy": cy,
#             "sensor_h": camera_data.height,
#             "sensor_w": camera_data.width,
#             "projection_mode": "perspective"
#         }
#         self.tensor_tran = lambda x:torch.from_numpy(x).to(torch.float32)
#     def __len__(self):
#         return len(self.img_files)

#     def __getitem__(self, index:int):
#         image = self.img_tran(Image.open(self.img_files[index]))
#         pcd = np.load(self.lidar_files[index])
#         pcd_rev = pcd[:,0] > 0
#         pcd = self.pcd_tran(pcd[pcd_rev,:])
#         pcd = self.resample_tran(pcd)  # (N, 3)
#         match_list = []
#         tgt_kpt_list = []
#         src_extran = self.cam_poses[index]
#         tgt_extran_list = []
#         for tgt_idx, matches in self.pair_dict[index]:
#             match_list.append(matches)
#             tgt_kpt_list.append(self.kpts[tgt_idx])
#             tgt_extran_list.append(self.cam_poses[tgt_idx])
#         intran = np.array([[self.camera_info['fx'],0,self.camera_info['cx']],
#                            [0,self.camera_info['fy'],self.camera_info['cy']],
#                            [0,0,1]])
#         image_hw = (self.camera_info['sensor_h'], self.camera_info['sensor_w'])
#         cba_data = dict(src_pcd=pcd, src_kpt=self.kpts[index], tgt_kpt_list=tgt_kpt_list, match_list=match_list,
#             src_extran=src_extran, tgt_extran_list=tgt_extran_list, intran=intran, scale=self.scale, img_hw=image_hw)
#         ca_data = dict(cam_mappoint=self.cam_mappoints[index], pcd=pcd, scale=self.scale)
#         return dict(img=image, pcd=self.tensor_tran(pcd.T), camera_info=self.camera_info, extran=self.tensor_tran(self.gt_Tcl), cba_data=cba_data, ca_data=ca_data)

# class FlowDataset(Dataset):
#     def __init__(self, gt_Tcl:str, image_dir:str, lidar_dir:str, lidar_pose:str, model_dir:str,
#              pair_file:str, flow_dir:str, filter_params:Optional[Dict[str,float]]=None, pcd_sample_num:int=-1
#              ):
#         self.gt_Tcl = np.loadtxt(gt_Tcl)
#         image_files = sorted(os.listdir(image_dir))
#         lidar_files = sorted(os.listdir(lidar_dir))
#         flow_files = sorted(os.listdir(flow_dir))
#         cameras, images, points3d = read_model(model_dir, '.bin')
#         camera_id = cameras.keys().__iter__().__next__()
#         camera_data = cameras[camera_id]
#         assert camera_data.model in CAMERA_TYPE.keys(), 'Unknown camera type:{}'.format(camera_data.model)
#         params_dict = {key: value for key, value in zip(CAMERA_TYPE[camera_data.model], camera_data.params)}
#         if 'f' in params_dict:
#             fx = fy = params_dict['f']
#         else:
#             fx = params_dict['fx']
#             fy = params_dict['fy']
#         cx = params_dict['cx']
#         cy = params_dict['cy']
#         assert len(images.keys()) == len(image_files) == len(lidar_files), "images ({}), image_files ({}), lidar_files ({})".format(len(images.keys()), len(image_files), len(lidar_files))
#         self.img_files = [os.path.join(image_dir, file) for file in image_files]
#         self.lidar_files = [os.path.join(lidar_dir, file) for file in lidar_files]
#         self.kpts = [np.load(os.path.join(kpt_dir, file))['keypoints'] for file in kpt_files]
#         matches = [np.load(os.path.join(match_dir, file))['match'] for file in sorted(os.listdir(match_dir))]
#         self.cam_poses = []
#         self.cam_mappoints = []
#         for i in range(1,len(image_files)+1):
#             img_data = images[i]
#             extrinsics = np.eye(4)
#             extrinsics[:3,:3] = img_data.qvec2rotmat()
#             extrinsics[:3,3] = img_data.tvec
#             self.cam_poses.append(extrinsics)
#             point3d_ids = img_data.point3D_ids
#             point3d_valid_mask = np.nonzero(point3d_ids != -1)[0]
#             point3d_ids = point3d_ids[point3d_valid_mask]
#             mapppoints = np.stack([points3d[idx].xyz for idx in point3d_ids], axis=0)
#             mapppoints = nptran(mapppoints, extrinsics)  # frame coordinate system
#             self.cam_mappoints.append(mapppoints)
#         pose_graph = o3d.io.read_pose_graph(lidar_pose)
#         lidar_pose = [inv_pose_np(node.pose) for node in pose_graph.nodes]
#         N = len(pose_graph.nodes)
#         lidar_edge = [src_pose @ inv_pose_np(tgt_pose) for src_pose, tgt_pose in zip(lidar_pose[:-1], lidar_pose[1:])]
#         camera_edge = [src_pose @ inv_pose_np(tgt_pose) for src_pose, tgt_pose in zip(self.cam_poses[:-1], self.cam_poses[1:])]
#         camera_rotedge, lidar_rotedge = map(lambda edge_list:[T[:3,:3] for T in edge_list],[camera_edge, lidar_edge])
#         ransac_state = dict(min_samples=3,
#                                 max_trials=5000,
#                                 stop_prob=0.99,
#                                 random_state=0)
#         ransac_rot_estimator = RotRANSAC(RotEstimator(),
#                                 **ransac_state)
#         alpha,beta = map(ransac_rot_estimator.toVecList,[camera_rotedge, lidar_rotedge])
#         best_rot, rot_inlier_mask = ransac_rot_estimator.fit(beta,alpha)
#         ransac_tsl_estimator = TslRANSAC(TslEstimator(best_rot),
#                                      **ransac_state)
#         inlier_camera_edge = [edge for i,edge in enumerate(camera_edge) if rot_inlier_mask[i]]
#         inlier_lidar_edge = [edge for i,edge in enumerate(lidar_edge) if rot_inlier_mask[i]]
#         camera_flatten = ransac_tsl_estimator.flatten(inlier_camera_edge)
#         lidar_flatten = ransac_tsl_estimator.flatten(inlier_lidar_edge)
#         _, best_scale, _ = ransac_tsl_estimator.fit(camera_flatten, lidar_flatten)
#         self.scale = best_scale
#         self.pair = json.load(open(pair_file, 'r'))
#         self.pair_dict = dict()
#         for i, pair in enumerate(self.pair):
#             src_idx, tgt_idx = pair
#             if tgt_idx - src_idx > max_frame_corr:
#                 continue
#             if src_idx not in self.pair_dict.keys():
#                 self.pair_dict[src_idx] = []
#             self.pair_dict[src_idx].append([tgt_idx, matches[i]])
#             if tgt_idx not in self.pair_dict.keys():
#                 self.pair_dict[tgt_idx] = []
#             self.pair_dict[tgt_idx].append([src_idx, matches[i][:,::-1]])  # swap source and target
#         self.img_tran = Tf.ToTensor()
#         self.image_shape = [camera_data.height, camera_data.width]
#         self.pcd_tran = KITTIFilter(**filter_params) if filter_params else lambda x:x
#         self.resample_tran = Resampler(pcd_sample_num)
#         self.camera_info = {
#             "fx": fx,
#             "fy": fy,
#             "cx": cx,
#             "cy": cy,
#             "sensor_h": camera_data.height,
#             "sensor_w": camera_data.width,
#             "projection_mode": "perspective"
#         }
#         self.tensor_tran = lambda x:torch.from_numpy(x).to(torch.float32)
#     def __len__(self):
#         return len(self.img_files)

#     def __getitem__(self, index:int):
#         image = self.img_tran(Image.open(self.img_files[index]))
#         pcd = np.load(self.lidar_files[index])
#         pcd_rev = pcd[:,0] > 0
#         pcd = self.pcd_tran(pcd[pcd_rev,:])
#         pcd = self.resample_tran(pcd)  # (N, 3)
#         match_list = []
#         tgt_kpt_list = []
#         src_extran = self.cam_poses[index]
#         tgt_extran_list = []
#         for tgt_idx, matches in self.pair_dict[index]:
#             match_list.append(matches)
#             tgt_kpt_list.append(self.kpts[tgt_idx])
#             tgt_extran_list.append(self.cam_poses[tgt_idx])
#         intran = np.array([[self.camera_info['fx'],0,self.camera_info['cx']],
#                            [0,self.camera_info['fy'],self.camera_info['cy']],
#                            [0,0,1]])
#         image_hw = (self.camera_info['sensor_h'], self.camera_info['sensor_w'])
#         cba_data = dict(src_pcd=pcd, src_kpt=self.kpts[index], tgt_kpt_list=tgt_kpt_list, match_list=match_list,
#             src_extran=src_extran, tgt_extran_list=tgt_extran_list, intran=intran, scale=self.scale, img_hw=image_hw)
#         ca_data = dict(cam_mappoint=self.cam_mappoints[index], pcd=pcd, scale=self.scale)
#         return dict(img=image, pcd=self.tensor_tran(pcd.T), camera_info=self.camera_info, extran=self.tensor_tran(self.gt_Tcl), cba_data=cba_data, ca_data=ca_data)

 
# class PertubSeqKITTIDataset(Dataset):
#     def __init__(self,dataset:BaseKITTIDataset,
#                  max_deg:float,
#                  max_tran:float,
#                  mag_randomly=True,
#                  ):
#         self.dataset = dataset
#         self.transform = transform.UniformTransformSE3(max_deg, max_tran, mag_randomly)

#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, index:Tuple[int, Sequence[int]]):
#         group_idx, sub_indices = index
#         igt_x = self.transform.generate_transform(1)  # (1, 6)
#         gt = se3.exp(-igt_x).squeeze(0)
#         igt = se3.exp(igt_x)
#         img = []
#         uncalib_pcd = []
#         for sub_idx in sub_indices:
#             data = self.dataset[(group_idx, sub_idx)]
#             _uncalib_pcd = se3.transform(igt, data['pcd'].unsqueeze(0))  # (1, 3, N)
#             img.append(data['img'])
#             uncalib_pcd.append(_uncalib_pcd)
#         new_data = dict(img=torch.stack(img, dim=0),uncalib_pcd=torch.cat(uncalib_pcd, dim=0), gt=gt, camera_info=data['camera_info'])
#         return new_data
    
#     @staticmethod
#     def collate_fn(zipped_x:Iterable[Dict[str, Union[torch.Tensor, Dict]]]):
#         batch = dict()
#         batch['img'] = torch.stack([x['img'] for x in zipped_x])
#         batch['uncalib_pcd'] = torch.stack([x['uncalib_pcd'] for x in zipped_x])
#         batch['gt'] = torch.stack([x['gt'] for x in zipped_x])
#         batch['camera_info'] = zipped_x[0]['camera_info']
#         return batch

# class KITTISeqBatchSampler(BatchSampler):
#     def __init__(self, num_sequences:int, len_of_sequences:Sequence[int], dataset_len:int, max_images:int=12,
#             num_samples_per_seq:Tuple[int,int]=(2,6), continuous:bool=True):
#         # Batch sampler with a dynamic number of sequences
#         # max_images >= number_of_sequences * images_per_sequence
#         assert num_sequences == len(len_of_sequences)
#         self.max_images = max_images
#         self.num_samples_per_seq = list(range(num_samples_per_seq[0],num_samples_per_seq[1]+1))
#         self.num_sequences = num_sequences
#         self.len_of_sequences = len_of_sequences
#         self.dataset_len = dataset_len
#         self.continuous = continuous

#     def __iter__(self):
#         for _ in range(self.dataset_len):
#             # number per sequence
#             num_sample = np.random.choice(self.num_samples_per_seq)
#             num_seq = self.max_images // num_sample
#             seq_idx = np.random.choice(self.num_sequences)
#             batches = []
#             for _ in range(num_seq):
#                 if not self.continuous:
#                     file_indices = np.random.choice(self.len_of_sequences[seq_idx], num_sample, replace=False)
#                 else:
#                     start_idx = np.random.choice(self.len_of_sequences[seq_idx] - num_sample)
#                     file_indices = np.arange(start_idx, start_idx+num_sample)
#                 batches.append((seq_idx, file_indices))
#             yield batches

#     def __len__(self):
#         return self.dataset_len
