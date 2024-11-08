import random
import numpy as np
import math
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from typing import Iterable, List, Tuple, Union
from models.util.constant import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import time
class CudaTimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_time = self.start_event.elapsed_time(self.end_event) / 1000 # ms -> s

class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        self.elapsed_time = time.time() - self.start_time

def patchidx(target_size:Iterable[int], patch_size:Iterable[int], overlap:Iterable[int]):
	Hindex = list(range(0, target_size[0] - overlap[0], patch_size[0] - overlap[0]))
	Windex = list(range(0, target_size[1] - overlap[1], patch_size[1] - overlap[1]))
	Hindex[-1] = target_size[0] - patch_size[0]
	Windex[-1] = target_size[1] - patch_size[1]
	return Hindex, Windex

def img2patch(img:torch.Tensor, patch_size:Iterable[int], overlap:Iterable[int]) -> List[torch.Tensor]:
	if len(img.shape) == 3:
		H, W = img.shape[1], img.shape[0]
	elif len(img.shape) == 2:
		H, W = img.shape[0], img.shape[1]
	Hindex, Windex = patchidx((H,W), patch_size, overlap)
	patches = []
	for hi in Hindex:
		for wi in Windex:
			patches.append(img[...,hi:hi+256, wi:wi+256])
	return patches

def patch2img(patches:List[torch.Tensor], Hindex:Iterable[int], Windex:Iterable[int], target_size:Tuple[int]) -> torch.Tensor:
	img = torch.zeros(target_size).to(patches[0])
	count = torch.zeros(img.shape[-2:]).to(img)
	for patch_idx, (hi,wi) in enumerate(zip(Hindex, Windex)):
		img[...,hi:hi+256, wi:wi+256] += patches[patch_idx]
		count[hi:hi+256, wi:wi+256] += 1
	return (img / count).to(img)
            

def tensor2img(tensor:torch.Tensor, out_type=np.uint8, mean_value:Union[float, Iterable]=IMAGENET_DEFAULT_MEAN, std_value:Union[float, Iterable]=IMAGENET_DEFAULT_STD):
	'''
	Converts a torch Tensor into an image Numpy array
	Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
	Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
	'''
	n_dim = tensor.dim()
	if isinstance(mean_value, Iterable):
		mean = np.array(mean_value)[None, None, :]  # (1,1,3)
		std = np.array(std_value)[None, None, :]  # (1,1,3)
	else:
		mean = mean_value
		std = std_value
	if n_dim == 4:
		n_img = len(tensor)
		img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).cpu().detach().numpy()
		img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
	elif n_dim == 3:
		img_np = tensor.cpu().detach().numpy()
		img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
	elif n_dim == 2:
		img_np = tensor.cpu().detach().numpy()
	else:
		raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
	if out_type == np.uint8:
		img_np = (img_np * std) + mean
		img_np = (np.clip(img_np, 0, 1) * 255).round()
		# Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
	return img_np.astype(out_type).squeeze()

def postprocess(images):
	return [tensor2img(image) for image in images]


def set_seed(seed, gl_seed=0):
	"""  set random seed, gl_seed used in worker_init_fn function """
	if seed >=0 and gl_seed>=0:
		seed += gl_seed
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)

	''' change the deterministic and benchmark maybe cause uncertain convolution behavior. 
		speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html '''
	if seed >=0 and gl_seed>=0:  # slower, more reproducible
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	else:  # faster, less reproducible
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True

def set_gpu(args):
	""" set parameter to gpu or ddp """
	if args is None:
		return None
	if isinstance(args, torch.nn.Module):
		return args.cuda()
		
def set_device(args):
	""" set parameter to gpu or cpu """
	if torch.cuda.is_available():
		if isinstance(args, list):
			return (set_gpu(item) for item in args)
		elif isinstance(args, dict):
			return {key:set_gpu(args[key]) for key in args}
		else:
			args = set_gpu(args)
	return args


def load_checkpoint(checkpoint:str, model:nn.Module, optimizer:torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler._LRScheduler):
    chkpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(chkpt['model'])
    optimizer.load_state_dict(chkpt['optimizer'])
    scheduler.load_state_dict(chkpt['scheduler'])
    last_epoch = chkpt['epoch']
    # scheduler.last_epoch = last_epoch
    best_loss = chkpt['best_loss']
    return last_epoch + 1, best_loss

def save_checkpoint(checkpoint:str, epoch:int, best_loss:float,
		model:nn.Module, optimizer:torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler._LRScheduler):
	torch.save(dict(model=model.state_dict(),
		optimizer=optimizer.state_dict(),
		scheduler=scheduler.state_dict(),
		epoch=epoch,
		best_loss=best_loss),
		checkpoint)
	
def load_checkpoint_model_only(checkpoint:str, model:nn.Module):
    chkpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(chkpt['model'])
    return