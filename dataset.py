import os
import json
import torch
from PIL import Image
from torchvision.transforms import transforms as Tf
import numpy as np
import pykitti
import open3d as o3d
from models.util import transform, se3
from PIL import Image
from torch import Generator, randperm
from torch.utils.data import Dataset, Subset, BatchSampler
from typing import Iterable, List, Dict, Union, Optional, Tuple, Sequence
from models.colmap.io import read_model, CAMERA_TYPE
from models.tools.cmsc import nptran
from models.util.transform import inv_pose_np
from RANSAC.base import RotEstimator,TslEstimator,RotRANSAC, TslRANSAC

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
    def __init__(self, voxel_size=0.3, min_dist:float=0.15):
        """KITTIFilter

        Args:
            voxel_size (float, optional): voxel size for downsampling. Defaults to 0.3.
            concat (str, optional): concat operation for normal estimation, 'none','xyz' or 'zero-mean'. Defaults to 'none'.
        """
        self.voxel_size = voxel_size
        self.min_dist = min_dist
        
    def __call__(self, x:np.ndarray):
        rev_x = np.linalg.norm(x, axis=1) > self.min_dist
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x[rev_x,:])
        # _, ind = pcd.remove_radius_outlier(nb_points=self.n_neighbor, radius=self.voxel_size)
        # pcd.select_by_index(ind)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        pcd_xyz = np.array(pcd.points,dtype=np.float32)
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
    

class KITTISeqBatchSampler(BatchSampler):
    def __init__(self, num_sequences:int, len_of_sequences:Sequence[int], dataset_len:int, max_images:int=12,
            num_samples_per_seq:Tuple[int,int]=(2,6), continuous:bool=True):
        # Batch sampler with a dynamic number of sequences
        # max_images >= number_of_sequences * images_per_sequence
        assert num_sequences == len(len_of_sequences)
        self.max_images = max_images
        self.num_samples_per_seq = list(range(num_samples_per_seq[0],num_samples_per_seq[1]+1))
        self.num_sequences = num_sequences
        self.len_of_sequences = len_of_sequences
        self.dataset_len = dataset_len
        self.continuous = continuous

    def __iter__(self):
        for _ in range(self.dataset_len):
            # number per sequence
            num_sample = np.random.choice(self.num_samples_per_seq)
            num_seq = self.max_images // num_sample
            seq_idx = np.random.choice(self.num_sequences)
            batches = []
            for _ in range(num_seq):
                if not self.continuous:
                    file_indices = np.random.choice(self.len_of_sequences[seq_idx], num_sample, replace=False)
                else:
                    start_idx = np.random.choice(self.len_of_sequences[seq_idx] - num_sample)
                    file_indices = np.arange(start_idx, start_idx+num_sample)
                batches.append((seq_idx, file_indices))
            yield batches

    def __len__(self):
        return self.dataset_len

class CBADataset(Dataset):
    def __init__(self, gt_Tcl:str, image_dir:str, lidar_dir:str, lidar_pose:str, model_dir:str,
             pair_file:str, kpt_dir:str, match_dir:str, max_frame_corr:int,
             filter_params:Optional[Dict[str,float]]=None, pcd_sample_num:int=-1
             ):
        self.gt_Tcl = np.loadtxt(gt_Tcl)
        image_files = sorted(os.listdir(image_dir))
        lidar_files = sorted(os.listdir(lidar_dir))
        kpt_files = sorted(os.listdir(kpt_dir))
        cameras, images, points3d = read_model(model_dir, '.bin')
        camera_id = cameras.keys().__iter__().__next__()
        camera_data = cameras[camera_id]
        assert camera_data.model in CAMERA_TYPE.keys(), 'Unknown camera type:{}'.format(camera_data.model)
        params_dict = {key: value for key, value in zip(CAMERA_TYPE[camera_data.model], camera_data.params)}
        if 'f' in params_dict:
            fx = fy = params_dict['f']
        else:
            fx = params_dict['fx']
            fy = params_dict['fy']
        cx = params_dict['cx']
        cy = params_dict['cy']
        assert len(images.keys()) == len(image_files) == len(lidar_files), "images ({}), image_files ({}), lidar_files ({})".format(len(images.keys()), len(image_files), len(lidar_files))
        self.img_files = [os.path.join(image_dir, file) for file in image_files]
        self.lidar_files = [os.path.join(lidar_dir, file) for file in lidar_files]
        self.kpts = [np.load(os.path.join(kpt_dir, file))['keypoints'] for file in kpt_files]
        matches = [np.load(os.path.join(match_dir, file))['match'] for file in sorted(os.listdir(match_dir))]
        self.cam_poses = []
        self.cam_mappoints = []
        for i in range(1,len(image_files)+1):
            img_data = images[i]
            extrinsics = np.eye(4)
            extrinsics[:3,:3] = img_data.qvec2rotmat()
            extrinsics[:3,3] = img_data.tvec
            self.cam_poses.append(extrinsics)
            point3d_ids = img_data.point3D_ids
            point3d_valid_mask = np.nonzero(point3d_ids != -1)[0]
            point3d_ids = point3d_ids[point3d_valid_mask]
            mapppoints = np.stack([points3d[idx].xyz for idx in point3d_ids], axis=0)
            mapppoints = nptran(mapppoints, extrinsics)  # frame coordinate system
            self.cam_mappoints.append(mapppoints)
        pose_graph = o3d.io.read_pose_graph(lidar_pose)
        lidar_pose = [inv_pose_np(node.pose) for node in pose_graph.nodes]
        N = len(pose_graph.nodes)
        lidar_edge = [src_pose @ inv_pose_np(tgt_pose) for src_pose, tgt_pose in zip(lidar_pose[:-1], lidar_pose[1:])]
        camera_edge = [src_pose @ inv_pose_np(tgt_pose) for src_pose, tgt_pose in zip(self.cam_poses[:-1], self.cam_poses[1:])]
        camera_rotedge, lidar_rotedge = map(lambda edge_list:[T[:3,:3] for T in edge_list],[camera_edge, lidar_edge])
        ransac_state = dict(min_samples=3,
                                max_trials=5000,
                                stop_prob=0.99,
                                random_state=0)
        ransac_rot_estimator = RotRANSAC(RotEstimator(),
                                **ransac_state)
        alpha,beta = map(ransac_rot_estimator.toVecList,[camera_rotedge, lidar_rotedge])
        best_rot, rot_inlier_mask = ransac_rot_estimator.fit(beta,alpha)
        ransac_tsl_estimator = TslRANSAC(TslEstimator(best_rot),
                                     **ransac_state)
        inlier_camera_edge = [edge for i,edge in enumerate(camera_edge) if rot_inlier_mask[i]]
        inlier_lidar_edge = [edge for i,edge in enumerate(lidar_edge) if rot_inlier_mask[i]]
        camera_flatten = ransac_tsl_estimator.flatten(inlier_camera_edge)
        lidar_flatten = ransac_tsl_estimator.flatten(inlier_lidar_edge)
        _, best_scale, _ = ransac_tsl_estimator.fit(camera_flatten, lidar_flatten)
        self.scale = best_scale
        self.pair = json.load(open(pair_file, 'r'))
        self.pair_dict = dict()
        for i, pair in enumerate(self.pair):
            src_idx, tgt_idx = pair
            if tgt_idx - src_idx > max_frame_corr:
                continue
            if src_idx not in self.pair_dict.keys():
                self.pair_dict[src_idx] = []
            self.pair_dict[src_idx].append([tgt_idx, matches[i]])
            if tgt_idx not in self.pair_dict.keys():
                self.pair_dict[tgt_idx] = []
            self.pair_dict[tgt_idx].append([src_idx, matches[i][:,::-1]])  # swap source and target
        self.img_tran = Tf.ToTensor()
        self.image_shape = [camera_data.height, camera_data.width]
        self.pcd_tran = KITTIFilter(**filter_params) if filter_params else lambda x:x
        self.resample_tran = Resampler(pcd_sample_num)
        self.camera_info = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "sensor_h": camera_data.height,
            "sensor_w": camera_data.width,
            "projection_mode": "perspective"
        }
        self.tensor_tran = lambda x:torch.from_numpy(x).to(torch.float32)
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index:int):
        image = self.img_tran(Image.open(self.img_files[index]))
        pcd = np.load(self.lidar_files[index])
        pcd_rev = pcd[:,0] > 0
        pcd = self.pcd_tran(pcd[pcd_rev,:])
        pcd = self.resample_tran(pcd)  # (N, 3)
        match_list = []
        tgt_kpt_list = []
        src_extran = self.cam_poses[index]
        tgt_extran_list = []
        for tgt_idx, matches in self.pair_dict[index]:
            match_list.append(matches)
            tgt_kpt_list.append(self.kpts[tgt_idx])
            tgt_extran_list.append(self.cam_poses[tgt_idx])
        intran = np.array([[self.camera_info['fx'],0,self.camera_info['cx']],
                           [0,self.camera_info['fy'],self.camera_info['cy']],
                           [0,0,1]])
        image_hw = (self.camera_info['sensor_h'], self.camera_info['sensor_w'])
        cba_data = dict(src_pcd=pcd, src_kpt=self.kpts[index], tgt_kpt_list=tgt_kpt_list, match_list=match_list,
            src_extran=src_extran, tgt_extran_list=tgt_extran_list, intran=intran, scale=self.scale, img_hw=image_hw)
        ca_data = dict(cam_mappoint=self.cam_mappoints[index], pcd=pcd, scale=self.scale)
        return dict(img=image, pcd=self.tensor_tran(pcd.T), camera_info=self.camera_info, extran=self.tensor_tran(self.gt_Tcl), cba_data=cba_data, ca_data=ca_data)

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


class PerturbDataset(Dataset):
    def __init__(self,dataset:Dataset,
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

    def __getitem__(self, index:int):
        data = self.dataset[index]
        if self.file is None:  # randomly generate igt
            igt_x = self.transform.generate_transform(1)
            igt = se3.exp(igt_x).squeeze(0)
            gt = transform.inv_pose(igt)
        else:
            igt = se3.exp(self.perturb[:,index,:]).squeeze(0)  # (1,6) -> (1,4,4)
            gt = transform.inv_pose(igt).squeeze(0)
        extran = igt @ data['extran']  # add noise to the ground-truth extran
        data.update(dict(gt=gt, extran=extran))  # extran here denotes the perturbed gt extran
        return data



class BaseKITTIDataset(Dataset):
    def __init__(self,basedir:str,
                 seqs:List[str]=['09','10'], cam_id:int=2,
                 meta_json='data_len.json', skip_frame=1,
                 voxel_size=0.15, min_dist=0.15, pcd_sample_num=8192,
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
        if self.resize_size is not None:
            self.img_tran = Tf.Compose([
                Tf.ToTensor(),
                Tf.Resize(self.resize_size)])
        else:
            self.img_tran = Tf.ToTensor()
        self.pcd_tran = KITTIFilter(voxel_size, min_dist)
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
        REVH,REVW = self.extend_ratio[0]*RH,self.extend_ratio[1] * RW
        K_cam_extend = K_cam.copy()  # K_cam_extend for dilated projection
        K_cam_extend[0,-1] *= self.extend_ratio[0]
        K_cam_extend[1,-1] *= self.extend_ratio[1]
        raw_img = raw_img.resize([RW,RH],Image.BILINEAR)
        _img = self.img_tran(raw_img)  # raw img input (3,H,W)
        pcd:np.ndarray = data.get_velo(sub_index)[:,:3]
        pcd = self.pcd_tran(pcd)
        calibed_pcd = nptran(pcd, T_cam2velo).T
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
        return dict(img=_img,pcd=_pcd, camera_info=camera_info, extran=T_cam2velo)
    
    @staticmethod
    def collate_fn(zipped_x:Iterable[Dict[str, Union[torch.Tensor, Dict]]]):
        batch = dict()
        batch['img'] = torch.stack([x['img'] for x in zipped_x])
        batch['pcd'] = torch.stack([x['pcd'] for x in zipped_x])
        batch['extran'] = torch.stack([x['extran'] for x in zipped_x])
        batch['camera_info'] = zipped_x[0]['camera_info']
        batch['camera_info']['fx'] = torch.tensor([x['camera_info']['fx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['fy'] = torch.tensor([x['camera_info']['fy'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cx'] = torch.tensor([x['camera_info']['cx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cy'] = torch.tensor([x['camera_info']['cy'] for x in zipped_x], dtype=torch.float32)
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
        new_data = dict(img=data['img'],pcd=data['pcd'], gt=gt, extran=extran, camera_info=data['camera_info'])
        return new_data
    
    @staticmethod
    def collate_fn(zipped_x:Iterable[Dict[str, Union[torch.Tensor, Dict]]]):
        batch = dict()
        batch['img'] = torch.stack([x['img'] for x in zipped_x])
        batch['pcd'] = torch.stack([x['pcd'] for x in zipped_x])
        batch['extran'] = torch.stack([x['extran'] for x in zipped_x])
        batch['gt'] = torch.stack([x['gt'] for x in zipped_x])
        batch['camera_info'] = zipped_x[0]['camera_info']
        batch['camera_info']['fx'] = torch.tensor([x['camera_info']['fx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['fy'] = torch.tensor([x['camera_info']['fy'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cx'] = torch.tensor([x['camera_info']['cx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cy'] = torch.tensor([x['camera_info']['cy'] for x in zipped_x], dtype=torch.float32)
        return batch
    
class PertubSeqKITTIDataset(Dataset):
    def __init__(self,dataset:BaseKITTIDataset,
                 max_deg:float,
                 max_tran:float,
                 mag_randomly=True,
                 ):
        self.dataset = dataset
        self.transform = transform.UniformTransformSE3(max_deg, max_tran, mag_randomly)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index:Tuple[int, Sequence[int]]):
        group_idx, sub_indices = index
        igt_x = self.transform.generate_transform(1)  # (1, 6)
        gt = se3.exp(-igt_x).squeeze(0)
        igt = se3.exp(igt_x)
        img = []
        uncalib_pcd = []
        for sub_idx in sub_indices:
            data = self.dataset[(group_idx, sub_idx)]
            _uncalib_pcd = se3.transform(igt, data['pcd'].unsqueeze(0))  # (1, 3, N)
            img.append(data['img'])
            uncalib_pcd.append(_uncalib_pcd)
        new_data = dict(img=torch.stack(img, dim=0),uncalib_pcd=torch.cat(uncalib_pcd, dim=0), gt=gt, camera_info=data['camera_info'])
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
    base_dataset = BaseKITTIDataset('data/kitti', seqs=['00','01'], skip_frame=3)
    dataset = PertubKITTIDataset(base_dataset, 30, 0.3, True)
    data = dataset[0]
    for key,value in data.items():
        if hasattr(value, 'shape'):
            shape = value.shape
        else:
            shape = value
        print('{key}: {shape}'.format(key=key, shape=shape))
    
        

        