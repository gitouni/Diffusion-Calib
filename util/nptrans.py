
import numpy as np
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
    
def toVec(SE3:np.ndarray):
    """SE3 Matrix to rvec and tvec

    Args:
        SE3 (np.ndarray): 4x4 `np.ndarray`

    Returns:
        rvec, tvec: `np.ndarray`
    """
    R = Rotation.from_matrix(SE3[:3,:3])
    rvec = R.as_rotvec()
    V = computeV(rvec)
    tvec = np.linalg.inv(V) @ SE3[:3,3]
    return rvec, tvec

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

def inv_pose(pose:np.ndarray):
    """inverse a SE(3) matrix

    Args:
        pose (np.ndarray): 4x4

    Returns:
        np.ndarray: 4x4
    """
    ivpose = np.eye(4)
    ivpose[:3,:3] = pose[:3,:3].T
    ivpose[:3,3] = -ivpose[:3,:3] @ pose[:3, 3]
    return ivpose