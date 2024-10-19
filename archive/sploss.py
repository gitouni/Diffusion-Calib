import numpy as np
from typing import List, Tuple, Optional, Iterable
import cv2
import os
from scipy.spatial import KDTree, cKDTree
import open3d as o3d
from models.tools.cmsc import nptran, project_constraint_corr_pts, project_corr_pts, toMat, toRMat, toVec, toVecR
from models.colmap.io import read_model, CAMERA_TYPE
from models.util.transform import inv_pose_np
from RANSAC.base import RotEstimator,TslEstimator,RotRANSAC, TslRANSAC


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    if np.ndim(wa) == 2:
        wa = wa[:, None]
        wb = wb[:, None]
        wc = wc[:, None]
        wd = wd[:, None]
    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def rmse(err:np.ndarray):
    return np.sqrt(np.sum(err**2, axis=-1))

def in_region_frame_test(pt_list:np.ndarray, region:np.ndarray):
    H, W = region.shape
    pt_int = np.array(pt_list,dtype=np.int32)
    rev = (pt_int[:,0] >= 0) * (pt_int[:,0] < W) * (pt_int[:,1] >= 0) * (pt_int[:,1] < H)
    rev_idx = np.arange(len(rev), dtype=np.int32)
    rev_idx = rev_idx[rev]
    rev[rev_idx] = region[pt_int[rev,1], pt_int[rev,0]]
    return rev

def in_frame_test(pt_list:np.ndarray, hw:Tuple[int,int]):
    H, W = hw
    rev = (pt_list[:,0] >= 0) * (pt_list[:,0] < W) * (pt_list[:,1] >= 0) * (pt_list[:,1] < H)
    return rev

def in_region_test(pt_list:np.ndarray, region:np.ndarray):
    pt_int = np.array(pt_list,dtype=np.int32)
    return region[pt_int[:,1], pt_int[:,0]]

def dist_pt_region(pt_list:np.ndarray, contours:List[np.ndarray], region:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute distances of point-to-region

    Args:
        pt_list (np.ndarray): (N, 2)
        contours (List[np.ndarray]): [[N, 2]]
        region (np.ndarray): (H, W)

    Returns:
        np.ndarray: mean error of point-to-region distance
    """
    pt_to_test = np.logical_not(in_region_test(pt_list, region))  # points outside the region
    if pt_to_test.sum() == 0:
        return 0, None
    # else:
    #     return pt_to_test.sum() / len(pt_list)
    pt_left = pt_list[pt_to_test]
    err = np.zeros(len(pt_left), dtype=np.float32)
    # img = cv2.imread('/home/ouni/CODE/Research/Vis-MVSNet/colmap/00/images_col/000001.png')
    # cv2.drawContours(img, contours, -1, (0,255,0), 3)
    # for pt in pt_list:
    #     cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (255,0,0), -1)
    for i,pt in enumerate(pt_left):
        min_dist = np.inf
        # cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0,0,255), -1)
        for contour in contours:
            dis = cv2.pointPolygonTest(contour, pt, True)
            # if dis < 0:
            #     min_dist = 0
            #     break
            min_dist = min(min_dist, abs(dis))
        err[i] = min_dist
    # cv2.imwrite('debug.png',img)
    return err, pt_left

def estimate_normal(pcd_arr:np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_arr)
    pcd.estimate_normals()
    return np.array(pcd.normals)

def dist2pt(pcd_arr:np.ndarray, pcd_norm:np.ndarray, pcd_tree:cKDTree, mappoint:np.ndarray, k:int=10, max_pt_err:float=0.5, max_norm_err:float=0.04, min_cnt:int=30):
    dist, ii = pcd_tree.query(mappoint, k=k, workers=-1)
    dist_rev = dist[:,0] < max_pt_err ** 2
    dist = dist[dist_rev]
    ii = ii[dist_rev]
    if len(ii) < min_cnt:
        return min_cnt - len(ii), None, None
    pcd_xyz_top1 = pcd_arr[ii[:,0]]  # (n, 3)
    pcd_norm_top1 = pcd_norm[ii[:,0]] # (n, 3)
    pcd_nn = pcd_arr[ii.reshape(-1)].reshape(ii.shape[0], ii.shape[1], 3)  # (n, k, 3)
    k = pcd_nn.shape[1]
    norm_reg = np.sum(np.abs((pcd_xyz_top1[:,None,:] - pcd_nn) * pcd_norm_top1[:,None,:]), axis=-1)  # (n, k)
    plane_rev = np.mean(norm_reg, axis=1) < max_norm_err
    err = np.where(plane_rev, np.abs(np.sum((mappoint[dist_rev] - pcd_xyz_top1) * pcd_norm_top1, axis=-1)), np.sqrt(dist[:, 0]))
    return err, dist_rev, ii[:,0]

def CBAError(lidar:np.ndarray, src_kpt:np.ndarray, tgt_kpt:np.ndarray, intran:np.ndarray, extran:np.ndarray, relpose:np.ndarray, img_shape:Tuple[int,int], min_cnt:int, max_dist:float, proj_constraint:bool=False) -> float:
    src_pcd = nptran(lidar, extran)  # camera coordinate (source frame)
    tgt_pcd = nptran(src_pcd, relpose)
    proj_func = project_constraint_corr_pts if proj_constraint else project_corr_pts
    src_proj, tgt_proj = proj_func(src_pcd, tgt_pcd, np.eye(4), intran, img_shape, return_indices=False)
    
    tree = KDTree(src_proj, leafsize=10)
    dist, ii = tree.query(src_kpt, k=1, eps=0.1)
    dist_rev = dist < max_dist ** 2
    ii = ii[dist_rev]
    if len(ii) < min_cnt:
        return min_cnt - len(ii)
    err = rmse(tgt_kpt[dist_rev] - tgt_proj[ii])
    return np.mean(err).item()

def CAError(pcd_arr:np.ndarray, pcd_norm:np.ndarray, pcd_tree:cKDTree, mappoint:np.ndarray, extran:np.ndarray, scale:float, min_cnt:int, k:int=10, max_pt_err:float=0.5, max_norm_err:float=0.04) -> float:
    transformed_mappoint = nptran(mappoint * scale, inv_pose_np(extran))
    err, _, _ = dist2pt(pcd_arr, pcd_norm, pcd_tree, transformed_mappoint, k, max_pt_err, max_norm_err, min_cnt)
    if isinstance(err, np.ndarray):
        return err.mean().item()
    return err

def SegError(lidar:np.ndarray, src_mask:np.ndarray, tgt_mask_list:List[np.ndarray], intran:np.ndarray, extran:np.ndarray, relpose_list:List[np.ndarray], img_shape:Tuple[int,int], min_cnt:int, src_imgs:Optional[np.ndarray]=None) -> np.ndarray:
    src_pcd = nptran(lidar, extran)  # camera coordinate (source frame)
    labels:list = np.unique(src_mask).tolist()
    if 0 in labels:
        labels.remove(0)
    src_region_list = [src_mask == l for l in labels]  # should exclude 0 (pixels without semantic information)
    err = np.zeros(len(tgt_mask_list))
    for fi, (tgt_mask, relpose) in enumerate(zip(tgt_mask_list, relpose_list)):
        tgt_pcd = nptran(src_pcd, relpose)  # camera coordinate (target frame)
        src_proj, tgt_proj = project_constraint_corr_pts(src_pcd, tgt_pcd, np.eye(4), intran, img_shape, return_indices=False)
        if len(src_proj) < min_cnt:
            err[fi] =  min_cnt - len(src_proj)   # not enough target reprojected points, ignore this frame
            continue
        label_err = np.empty(0, dtype=np.float32)
        for label, src_region in zip(labels, src_region_list):
            proj_rev = in_region_test(src_proj, src_region)
            if proj_rev.sum() == 0:
                continue
            tgt_region = tgt_mask == label
            if tgt_region.sum() == 0:  # no target mask, ignore this mask region
                continue
            tgt_region_img = tgt_region.astype(np.uint8) * 255
            tgt_contours, _ = cv2.findContours(tgt_region_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            tgt_contour_set = [np.squeeze(contour, axis=1) for contour in tgt_contours]  # (N, 1, 2) -> (N, 2)
            dist, pt_left = dist_pt_region(tgt_proj[proj_rev], tgt_contour_set, tgt_region)
            if src_imgs is not None and pt_left is not None:
                for pt in pt_left:
                    cv2.circle(src_imgs[fi], (int(pt[0]), int(pt[1])), 3, (0,0,255),1)
            label_err = np.hstack((label_err, dist))
        err[fi] = label_err.mean()
    return err

class CMSCLoss:
    def __init__(self,
            lidar_dir:str,
            lidar_pose:str,
            kpt_dir:str,
            kpt_match_dir:str,
            colmap_model_dir:str,
            max_frame_dist:int,
            cba_min_cnt:int,
            ca_min_cnt:int,
            mutual_match:bool,
            proj_constraint:bool,
            cba_max_corr_dist:float = 2.0,
            ca_knn:int = 10,
            ca_max_corr_dist:float = 0.5,
            norm_reg_err:float = 0.04,
            debug:bool = False,
            NonLieGroup:bool = False,
            place_holder:Optional[Iterable]=None,
        ):
        self.lidar_files = [os.path.join(lidar_dir, file) for file in sorted(os.listdir(lidar_dir))]
        self.kpt_files = [os.path.join(kpt_dir, file) for file in sorted(os.listdir(kpt_dir))]
        self.match_files = [os.path.join(kpt_match_dir, file) for file in sorted(os.listdir(kpt_match_dir))]
        self.max_frame_dist = max_frame_dist
        self.cba_min_cnt = cba_min_cnt
        self.ca_min_cnt = ca_min_cnt
        self.mutual_match = mutual_match
        self.proj_constraint = proj_constraint
        self.cba_max_corr_dist = cba_max_corr_dist
        self.ca_knn = ca_knn
        self.ca_max_corr_dist = ca_max_corr_dist
        self.norm_reg_err = norm_reg_err
        self.debug = debug
        files = os.listdir(colmap_model_dir)
        if 'images.bin' in files:
            ext = '.bin'
        elif 'images.txt' in files:
            ext = '.txt'
        else:
            raise FileNotFoundError('No valid files in colmap model dir:{}'.format(colmap_model_dir))
        self.pcds = []
        self.pcd_normals = []
        self.pcd_trees = []
        if debug:
            print('loading lidar points...')
        for lidar_file in self.lidar_files:
            pcd = np.load(lidar_file)
            pcd_rev = pcd[:,0] > 0
            pcd = pcd[pcd_rev]
            pcd_normal = estimate_normal(pcd)
            pcd_tree = KDTree(pcd, leafsize=100)
            self.pcds.append(pcd)
            self.pcd_normals.append(pcd_normal)
            self.pcd_trees.append(pcd_tree)
        if debug:
            print('loading camera 3D points...')
        cameras, images, points3d = read_model(colmap_model_dir, ext)
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
        self.intran = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,0,1]
        ])
        self.img_shape = cameras[camera_id].height, cameras[camera_id].width
        self.cam_mappoints = []
        self.cam_extran = []
        for i in range(1,len(images)+1):
            image = images[i]
            extrinsics = np.eye(4)
            extrinsics[:3,:3] = image.qvec2rotmat()
            extrinsics[:3,3] = image.tvec
            self.cam_extran.append(extrinsics)  # Tcw
            point3d_ids = image.point3D_ids
            point3d_valid_mask = np.nonzero(point3d_ids != -1)[0]
            point3d_ids = point3d_ids[point3d_valid_mask]
            mapppoints = np.stack([points3d[idx].xyz for idx in point3d_ids], axis=0)
            mapppoints = nptran(mapppoints, extrinsics)  # frame coordinate system
            self.cam_mappoints.append(mapppoints)
        if debug:
            print('loading image keypoints...')
        self.kpts = []
        for key_file in self.kpt_files:
            kpt_data = np.load(key_file)
            self.kpts.append(kpt_data['keypoints'])  # (N, 2) only the coordinates
        if debug:
            print('loading image keypoint matches...')
        self.match_dict = dict()
        for match_file in self.match_files:
            match_data = np.load(match_file)
            src_idx = match_data['src_idx'].item()
            tgt_idx = match_data['tgt_idx'].item()
            if abs(tgt_idx - src_idx) > self.max_frame_dist:
                continue
            corr = match_data['match']
            if src_idx not in self.match_dict.keys():
                self.match_dict[src_idx] = dict()
            self.match_dict[src_idx][tgt_idx] = corr # (N, 2): col1 source indices, col2 target indices
            if self.mutual_match:
                if tgt_idx not in self.match_dict.keys():
                    self.match_dict[tgt_idx] = dict()
                self.match_dict[tgt_idx][src_idx] = corr[:,::-1] # swap the source and target indices
        pose_graph = o3d.io.read_pose_graph(lidar_pose)
        lidar_pose = [inv_pose_np(node.pose) for node in pose_graph.nodes]
        lidar_edge = [src_pose @ inv_pose_np(tgt_pose) for src_pose, tgt_pose in zip(lidar_pose[:-1], lidar_pose[1:])]
        camera_edge = [src_pose @ inv_pose_np(tgt_pose) for src_pose, tgt_pose in zip(self.cam_extran[:-1], self.cam_extran[1:])]
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
        self.toMat = toRMat if NonLieGroup else toMat
        self.toVec = toVecR if NonLieGroup else toVec
        if place_holder is not None:
            self.place_holder = np.array(place_holder, dtype=np.bool_)
        else:
            self.place_holder = None

    def __call__(self, x0:np.ndarray, debug:Optional[bool]=None, x0_ref:Optional[np.ndarray]=None):
        def cba_func(src_idx:int, src_matched_data:np.ndarray) -> float:
            src_kpt = self.kpts[src_idx]
            src_lidar = self.pcds[src_idx]
            inv_src_cam_extran = inv_pose_np(self.cam_extran[src_idx])
            for tgt_idx, corr in src_matched_data.items():
                tgt_kpt = self.kpts[tgt_idx]
                src_kpt_corr = src_kpt[corr[:,0]]
                tgt_kpt_corr = tgt_kpt[corr[:,1]]
                relpose = self.cam_extran[tgt_idx] @ inv_src_cam_extran
                relpose[:3, 3] *= scale
                cba_err = CBAError(src_lidar, src_kpt_corr, tgt_kpt_corr, self.intran, extran, relpose, self.img_shape, self.cba_min_cnt, self.cba_max_corr_dist, self.proj_constraint)
                return cba_err
        def ca_func(cam_mappoint:np.ndarray, pcd:np.ndarray, pcd_norm:np.ndarray, pcd_tree:cKDTree):
            ca_err = CAError(pcd, pcd_norm, pcd_tree, cam_mappoint, extran, scale, self.ca_min_cnt, self.ca_knn, self.ca_max_corr_dist, self.norm_reg_err)
            return ca_err
        
        if debug is not None:
            self.debug = debug
        if self.place_holder is None:
            extran = self.toMat(x0[:3],x0[3:6])
        else:
            assert x0_ref is not None
            x0_ = x0_ref.copy()
            x0_[self.place_holder] = x0
            extran = self.toMat(x0_[:3],x0_[3:6])
        scale = self.scale  # fixed during the optimization
        cba_err_list = []
        ca_err_list = []
        for src_idx, src_matched_data in self.match_dict.items():
            cba_err_list.append(cba_func(src_idx, src_matched_data))
        for cam_mappoint, pcd, pcd_norm, pcd_tree in zip(self.cam_mappoints, self.pcds, self.pcd_normals, self.pcd_trees):
            ca_err_list.append(ca_func(cam_mappoint, pcd, pcd_norm, pcd_tree))
        cba_err = np.mean(np.array(cba_err_list))
        ca_err = np.mean(np.array(ca_err_list))
        return cba_err, ca_err

    def weighted_loss(self, x0:np.ndarray, weights:Tuple[float,float], x0_ref:Optional[np.ndarray]=None):
        cba_err, ca_err = self(x0, x0_ref=x0_ref)
        return cba_err * weights[0] + ca_err * weights[1]
    
class FLowLoss:
    def __init__(self,
            flow_dir:str,
            NonLieGroup:bool = False,
            max_err:float = 15.0,
            place_holder:Optional[Iterable]=None):
        self.toMat = toRMat if NonLieGroup else toMat
        self.toVec = toVecR if NonLieGroup else toVec
        if place_holder is not None:
            self.place_holder = np.array(place_holder, dtype=np.bool_)
        else:
            self.place_holder = None
        flow_files = sorted(os.listdir(flow_dir))
        self.src_pcd = []
        self.tgt_pcd = []
        self.intran = []
        self.optflow = []
        self.max_err = max_err
        for flow_file in flow_files:
            data = np.load(os.path.join(flow_dir, flow_file))
            self.src_pcd.append(data['src_pcd'])
            self.tgt_pcd.append(data['tgt_pcd'])
            self.intran.append(data['intran'])
            self.optflow.append(data['optflow'])

    def __call__(self, x0:np.ndarray, x0_ref:Optional[np.ndarray]=None):
        def floww_func(src_pcd:np.ndarray, tgt_pcd:np.ndarray, optflow:np.ndarray, intran:np.ndarray, extran:np.ndarray) -> float:
            img_shape = optflow.shape[-2:]
            src_proj, tgt_proj = project_constraint_corr_pts(src_pcd, tgt_pcd, extran, intran, img_shape)
            flow = tgt_proj - src_proj
            optflow_x = bilinear_interpolate(optflow[0,...], src_proj[:,0], src_proj[:,1])
            optflow_y = bilinear_interpolate(optflow[1,...], src_proj[:,0], src_proj[:,1])
            err = np.sqrt((flow[:,0]-optflow_x)**2 + (flow[:,1]-optflow_y)**2)
            err[err > self.max_err] = self.max_err
            return np.mean(err)
        if self.place_holder is None:
            extran = self.toMat(x0[:3],x0[3:6])
        else:
            assert x0_ref is not None
            x0_ = x0_ref.copy()
            x0_[self.place_holder] = x0
            extran = self.toMat(x0_[:3],x0_[3:6])
        err_list = []
        for src_pcd, tgt_pcd, intran, optflow in zip(self.src_pcd, self.tgt_pcd, self.intran, self.optflow):
            err = floww_func(src_pcd, tgt_pcd, optflow, intran, extran)
            err_list.append(err.item())
        return sum(err_list) / len(err_list)