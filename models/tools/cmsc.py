from .utils import k_nearest_neighbor, project_pc2image
import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, Iterable
from scipy.spatial.transform import Rotation

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

def CBABatchCorr(src_pcd:np.ndarray, src_kpt:np.ndarray, match_list:Iterable[np.ndarray],
        src_extran:np.ndarray, tgt_extran_list:Iterable[np.ndarray],
        intran:np.ndarray, Tcl:np.ndarray, scale:float, img_hw:Tuple[int,int],
        max_dist:float, proj_constraint:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute CBA Correspondences

    Args:
        src_pcd (np.ndarray): source point cloud (N, 3)
        src_kpt (np.ndarray): source keypoints (M, 2)
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
        Tuple[np.ndarray, np.ndarray]: index of src_pcd, index of src_kpt/tgt_kpt 
    """
    src_pcd = nptran(src_pcd, Tcl)  # camera coordinate (source frame)
    src_proj, src_rev = npproj(src_pcd, np.eye(4), intran, img_hw)
    
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
