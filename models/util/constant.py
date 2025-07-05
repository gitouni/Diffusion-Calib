from typing import List, Literal, TypedDict
import torch
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

__all__ = ['CameraInfoDict', 'BatchedCameraInfoDict', 'BaseDatasetOutput', 'BatchedBaseDatasetOutput', 'PerturbDatasetOutput', 'BatchedPerturbDatasetOutput']

class CameraInfoDict(TypedDict):
    fx: float
    fy: float
    cx: float
    cy: float
    sensor_h: int
    sensor_w: int
    projection_mode: Literal['perspective','parallel']

class BatchedCameraInfoDict(CameraInfoDict):
    fx: torch.Tensor
    fy: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor

class BaseDatasetOutput(TypedDict):
    img: torch.Tensor
    pcd: torch.Tensor
    camera_info: CameraInfoDict
    extran: torch.Tensor
    group_idx: int
    sub_idx: int

class BatchedBaseDatasetOutput(BaseDatasetOutput):
    camera_info: BatchedCameraInfoDict
    group_idx: List[int]
    sub_idx: List[int]

class PerturbDatasetOutput(TypedDict):
    img: torch.Tensor
    pcd: torch.Tensor
    gt: torch.Tensor
    extran: torch.Tensor
    pose_target: torch.Tensor
    camera_info: CameraInfoDict
    group_idx: int
    sub_dix: int

class BatchedPerturbDatasetOutput(PerturbDatasetOutput):
    camera_info: BatchedCameraInfoDict
    group_idx: List[int]
    sub_idx: List[int]