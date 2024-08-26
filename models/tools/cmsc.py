import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, Iterable, List, Dict
from scipy.spatial.transform import Rotation
import open3d as o3d
from scipy.spatial import KDTree, cKDTree

def inv_pose_np(pose_mat:np.ndarray):
    inv_pose_mat = pose_mat.copy()
    inv_pose_mat[...,:3,:3] = pose_mat[...,:3,:3].transpose(-1,-2)
    inv_pose_mat[...,:3,[3]] = -inv_pose_mat[...,:3,:3] @ pose_mat[...,:3,[3]]
    return inv_pose_mat

def skew(x:np.ndarray):
    return np.array([[0,-x[2],x[1]],
                     [x[2],0,-x[0]],
                     [-x[1],x[0],0]])
    
def computeV(rvec:np.ndarray):
    theta = np.linalg.norm(rvec)
    skew_rvec = skew(rvec)
    skew_rvec2 = skew_rvec @ skew_rvec
    if theta > 1e-8:
        V = np.eye(3) + (1 - np.cos(theta))/theta**2 * skew_rvec + (theta - np.sin(theta))/theta**3 * skew_rvec2
    else:
        V = np.eye(3) + (0.5 - 1/24 * theta**2) * skew_rvec + (1/6 - 1/120*theta**2) * skew_rvec2
    return V


def toVecSplit(rot:np.ndarray, tsl:np.ndarray):
    """SE3 Matrix to rvec and tvec

    Args:
        rot (np.ndarray): 3x3 `np.ndarray`
        tsl (np.ndarray): 3 `np.ndarray`

    Returns:
        rvec, tvec: `np.ndarray`
    """
    R = Rotation.from_matrix(rot)
    rvec = R.as_rotvec()
    V = computeV(rvec)
    tvec = np.linalg.inv(V) @ tsl
    return rvec, tvec

def toVec(SE3:np.ndarray):
    """SE3 Matrix to rvec and tvec

    Args:
        SE3 (np.ndarray): 4x4 `np.ndarray`

    Returns:
        rvec, tvec: `np.ndarray`
    """
    return toVecSplit(SE3[:3,:3], SE3[:3,3])

def toVecRSplit(rot:np.ndarray, tsl:np.ndarray):
    """SE3 Matrix to rvec and tvec

    Args:
        rot (np.ndarray): 3x3 `np.ndarray`
        tsl (np.ndarray): 3 `np.ndarray`

    Returns:
        rvec, tvec: `np.ndarray`
    """
    R = Rotation.from_matrix(rot)
    rvec = R.as_rotvec()
    return rvec, tsl

def toVecR(SE3:np.ndarray):
    """SE3 Matrix to rvec and tvec

    Args:
        SE3 (np.ndarray): 4x4 `np.ndarray`

    Returns:
        rvec, tvec: `np.ndarray`
    """
    return toVecRSplit(SE3[:3,:3], SE3[:3,3])

def toMatTotal(vec:np.ndarray):
    return toMat(vec[...,:3], vec[...,3:])

def toMat(rvec:np.ndarray, tvec:np.ndarray):
    """rvec and tvec to SE3 Matrix

    Args:
        rvec (`np.ndarray`): 1x3 rotation vector\n
        tvec (`np.ndarray`): 1x3 translation vector
        
    Returns:
        SE3: 4x4 `np.ndarray`
    """
    R = Rotation.from_rotvec(rvec)
    V = computeV(rvec)
    mat = np.eye(4)
    mat[:3,:3] = R.as_matrix()
    mat[:3,3] = V @ tvec
    return mat


def toRMat(rvec:np.ndarray, tsl:np.ndarray):
    """rvec and tvec to SE3 Matrix

    Args:
        rvec (`np.ndarray`): 1x3 rotation vector\n
        tsl (`np.ndarray`): 1x3 translation
        
    Returns:
        SE3: 4x4 `np.ndarray`
    """
    R = Rotation.from_rotvec(rvec)
    mat = np.eye(4)
    mat[:3,:3] = R.as_matrix()
    mat[:3,3] = tsl
    return mat

def inv_pose(pose:np.ndarray):
    ivpose = np.eye(4)
    ivpose[:3,:3] = pose[:3,:3].T
    ivpose[:3,3] = -ivpose[:3,:3] @ pose[:3, 3]
    return ivpose

def nptran(pcd:np.ndarray, rigdtran:np.ndarray) -> np.ndarray:
    pcd_ = pcd.copy().T  # (N, 3) -> (3, N)
    pcd_ = rigdtran[:3, :3] @ pcd_ + rigdtran[:3, [3]]
    return pcd_.T

def npproj(pcd:np.ndarray, extran:np.ndarray, intran:np.ndarray, img_shape:tuple, return_depth=False):
    """_summary_

    Args:
        pcd (np.ndarray): Nx3\\
        extran (np.ndarray): 4x4\\
        intran (np.ndarray): 3x3\\
        img_shape (tuple): HxW\\

    Returns:
        _type_: uv (N,2), rev (N,)
    """
    H, W = img_shape[0], img_shape[1]
    pcd_ = nptran(pcd, extran)  # (N, 3)
    if intran.shape[1] == 4:
        proj_pcd = intran @ np.concatenate([pcd_, np.ones([pcd_.shape[0],1])],axis=1).T
    else:
        proj_pcd = intran @ pcd_.T  # (3, N)
    u, v, w = proj_pcd[0], proj_pcd[1], proj_pcd[2]
    raw_index = np.arange(u.size)
    rev = w > 0
    raw_index = raw_index[rev]
    u = u[rev]/w[rev]
    v = v[rev]/w[rev]
    rev2 = (0<=u) * (u<W-1) * (0<=v) * (v<H-1)
    proj_pts = np.stack((u[rev2],v[rev2]),axis=1)
    if return_depth:
        return proj_pts, raw_index[rev2], pcd_[rev][rev2, 2]
    else:
        return proj_pts, raw_index[rev2]  # (N, 2), (N,)


def npproj_wocons(pcd:np.ndarray, extran:np.ndarray, intran:np.ndarray):
    """_summary_

    Args:
        pcd (np.ndarray): Nx3\\
        extran (np.ndarray): 4x4\\
        intran (np.ndarray): 3x3\\

    Returns:
        _type_: uv (N,2), rev (N,)
    """
    pcd_ = nptran(pcd, extran)  # (N, 3)
    if intran.shape[1] == 4:
        pcd_ = intran @ np.concatenate([pcd_, np.ones([pcd_.shape[0],1])],axis=1).T
    else:
        pcd_ = intran @ pcd_.T  # (3, N)
    u, v, w = pcd_[0], pcd_[1], pcd_[2]
    u = u/w
    v = v/w
    return np.stack((u,v),axis=1)  # (N, 2), (N,)

def project_corr_pts(src_pcd_corr:np.ndarray, tgt_pcd_corr:np.ndarray, extran:np.ndarray, intran:np.ndarray, img_shape:tuple, toint32:bool=False, return_indices:bool=False):
    src_proj_pts, src_rev_idx = npproj(src_pcd_corr, extran, intran, img_shape)
    tgt_proj_pts = npproj_wocons(tgt_pcd_corr[src_rev_idx], extran, intran)
    if toint32:
        src_proj_pts = src_proj_pts.astype(np.int32)
        tgt_proj_pts = tgt_proj_pts.astype(np.int32)
    if return_indices:
        return src_proj_pts, tgt_proj_pts, src_rev_idx
    else:
        return src_proj_pts, tgt_proj_pts
    
def project_constraint_corr_pts(src_pcd_corr:np.ndarray, tgt_pcd_corr:np.ndarray, extran:np.ndarray, intran:np.ndarray, img_shape:tuple, toint32:bool=False, return_indices:bool=False):
    src_proj_pts, src_rev_idx = npproj(src_pcd_corr, extran, intran, img_shape)
    tgt_proj_pts, tgt_rev_idx = npproj(tgt_pcd_corr, extran, intran, img_shape)
    _, src_inter_rev, tgt_inter_rev = np.intersect1d(src_rev_idx, tgt_rev_idx, assume_unique=True, return_indices=True)
    src_proj_pts = src_proj_pts[src_inter_rev]
    tgt_proj_pts = tgt_proj_pts[tgt_inter_rev]
    if toint32:
        src_proj_pts = src_proj_pts.astype(np.int32)
        tgt_proj_pts = tgt_proj_pts.astype(np.int32)
    if return_indices:
        return src_proj_pts, tgt_proj_pts, src_rev_idx[src_inter_rev]
    else:
        return src_proj_pts, tgt_proj_pts


def estimate_normal(pcd_arr:np.ndarray, radius:float=0.6, knn:int=15):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_arr)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius, knn))
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

def CBACorr(src_pcd:np.ndarray, src_kpt:np.ndarray,
        src_extran:np.ndarray, tgt_extran:np.ndarray,
        intran:np.ndarray, Tcl:np.ndarray, scale:float, img_hw:Tuple[int,int],
        max_dist:float, proj_constraint:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute CBA Correspondences

    Args:
        src_pcd (np.ndarray): source point cloud (N, 3)
        src_kpt (np.ndarray): source keypoints (M, 2)
        src_extran (np.ndarray): Tcw of source frame (4,4)
        tgt_extran (np.ndarray): Tcw of target frame (4,4)
        intran (np.ndarray): intrinsic matrix (3,3)
        Tcl (np.ndarray): extrinsic matrix from lidar to camera (4,4)
        scale (float): scale factor from camera to lidar
        img_hw (Tuple[int,int]): image shape: H, W
        max_dist (float): maximum distance to build CBA correspondences
        proj_constraint (bool, optional): whether to apply image bouding constraints to cross-frame projection. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: index of src_pcd, index of src_kpt/tgt_kpt 
    """
    src_pcd = nptran(src_pcd, Tcl)  # camera coordinate (source frame)
    relpose = src_extran @ inv_pose(tgt_extran)
    relpose[:3,3] *= scale
    tgt_pcd = nptran(src_pcd, relpose)
    proj_func = project_constraint_corr_pts if proj_constraint else project_corr_pts
    src_proj, _, src_rev = proj_func(src_pcd, tgt_pcd, np.eye(4), intran, img_hw, return_indices=True)
    tree = KDTree(src_proj, leafsize=10)
    dist, ii = tree.query(src_kpt, k=1, eps=0.1)
    dist_rev = dist < max_dist ** 2
    ii = ii[dist_rev]
    return src_rev[ii], dist_rev

def CBABatchCorr(src_pcd:np.ndarray, src_kpt:np.ndarray, 
        tgt_kpt_list:Iterable[np.ndarray], match_list:Iterable[np.ndarray],
        src_extran:np.ndarray, tgt_extran_list:Iterable[np.ndarray],
        intran:np.ndarray, Tcl:np.ndarray, scale:float, img_hw:Tuple[int,int],
        max_dist:float, proj_constraint:bool=False) -> List[Dict[str, np.ndarray]]:
    """Compute CBA Correspondences

    Args:
        src_pcd (np.ndarray): source point cloud (N, 3)
        src_kpt (np.ndarray): source keypoints (M1, 2)
        tgt_kpt_list:Iterable[np.ndarray]: Iterable of target indices keypoints: (M2, 2)
        match_list (Iterable[np.ndarray]): Iterable of match indices format:(src_idx, tgt_idx)
        src_extran (np.ndarray): Tcw of source frame (4,4)
        tgt_extran_list (Iterable[np.ndarray]): Sequence of Tcw of target frame (4,4)
        intran (np.ndarray): intrinsic matrix (3,3)
        Tcl (np.ndarray): extrinsic matrix from lidar to camera (4,4)
        scale (float): scale factor from camera to lidar
        img_hw (Tuple[int,int]): image shape: H, W
        max_dist (float): maximum distance to build CBA correspondences
        proj_constraint (bool, optional): whether to apply image bouding constraints to cross-frame projection. Defaults to False.

    Returns:
        List[Dict[str, np.ndarray]]: src_pcd, tgt_kpt, relpose
    """
    proj_func = project_constraint_corr_pts if proj_constraint else project_corr_pts
    raw_src_pcd = nptran(src_pcd, Tcl)  # camera coordinate (source frame)
    corr_data = []
    inv_src_extran = inv_pose(src_extran)
    for match, tgt_kpt, tgt_extran in zip(match_list, tgt_kpt_list, tgt_extran_list):
        relpose = tgt_extran @ inv_src_extran  # Tc1,w x Tw,c2
        relpose[:3,3] *= scale
        tgt_pcd = nptran(raw_src_pcd, relpose)
        src_proj, _, src_proj_rev = proj_func(raw_src_pcd, tgt_pcd, np.eye(4), intran, img_hw, return_indices=True)
        tree = KDTree(src_proj, leafsize=10)
        dist, ii = tree.query(src_kpt[match[:,0], :], k=1)  # len of src_kpt
        dist_rev = dist < max_dist ** 2
        if dist_rev.sum() == 0:
            continue  # skip this data
        ii = ii[dist_rev]  # indices of src_pcd[src_rev]
        matched_tgt_kpt = tgt_kpt[match[:,1],:][dist_rev]
        matched_src_pcd = src_pcd[src_proj_rev][ii]
        corr_data.append(dict(src_pcd=matched_src_pcd,
            tgt_kpt=matched_tgt_kpt, relpose=relpose))
    return corr_data

def CABatchCorr(cam_mappoint:np.ndarray, pcd:np.ndarray,
                 Tcl:np.ndarray, scale:float, max_dist:float,
                 normal_radius:float=0.6, noraml_knn:int=10, ca_knn:int = 10, norm_reg_err:float = 0.04):
    transformed_mappoint = nptran(cam_mappoint * scale, inv_pose_np(Tcl))
    pcd_norm = estimate_normal(pcd, normal_radius, noraml_knn)
    pcd_tree = KDTree(pcd, leafsize=100)
    dist, ii = pcd_tree.query(transformed_mappoint, k=ca_knn)
    dist_rev = dist[:,0] < max_dist ** 2
    dist = dist[dist_rev]
    ii = ii[dist_rev]
    pcd_xyz_top1 = pcd[ii[:,0]]  # (n, 3)
    pcd_norm_top1 = pcd_norm[ii[:,0]] # (n, 3)
    pcd_nn = pcd[ii.reshape(-1)].reshape(ii.shape[0], ii.shape[1], 3)  # (n, k, 3)
    norm_reg = np.sum(np.abs((pcd_xyz_top1[:,None,:] - pcd_nn) * pcd_norm_top1[:,None,:]), axis=-1)  # (n, k)
    plane_rev = np.mean(norm_reg, axis=1) < norm_reg_err
    corr_data = dict()
    corr_data['src_pt_pcd'] = pcd_xyz_top1[~plane_rev]
    corr_data['src_pt_campt'] = cam_mappoint[dist_rev][~plane_rev] * scale  # scaled camera mappoints
    corr_data['src_pl_pcd'] = pcd_xyz_top1[plane_rev]
    corr_data['src_pl_norm'] = pcd_norm_top1[plane_rev]
    corr_data['src_pl_campt'] = cam_mappoint[dist_rev][plane_rev] * scale  # scaled camera mappoints
    return corr_data