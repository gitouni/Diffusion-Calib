import math
import torch
import torch.nn as nn
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from typing import Union, Tuple, Literal, Iterable, Dict, Callable, Optional
from .util import se3
from .denoiser import Denoiser
from .dpm import NoiseScheduleVP, DPM_Solver, model_wrapper
from .tools.cmsc import CBABatchCorr
from .tools.utils import project_pc2image

def exists(x):
	return x is not None

def default(val, d):
	if exists(val):
		return val
	return d() if isfunction(d) else d

def extract(a:torch.Tensor, t:torch.Tensor, x_shape=(1,1,1,1)):
	b, *_ = t.shape
	out = a.gather(-1, t)
	return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
	scale = 1000 / timesteps
	beta_start = scale * 0.0001
	beta_end = scale * 0.02
	return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
	"""
	cosine schedule
	as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
	"""
	steps = timesteps + 1
	x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
	alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
	alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
	betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
	return torch.clip(betas, 0, 0.999)

def make_beta_schedule(schedule:Literal['quad','linear','warmup10','warmup50','const','jsd','cosine'],
						n_timestep:int, linear_start:float=1e-6, linear_end:float=1e-2, cosine_s:float=8e-3):
	"""beta schedule

	Args:
		schedule (str): Literal['quad','linear','warmup10','warmup50','const','jsd','cosine']
		n_timestep (int): number of timesteps
		linear_start (float, optional): Defaults to 1e-6.
		linear_end (float, optional): Defaults to 1e-2.
		cosine_s (float, optional): Defaults to 8e-3.

	Raises:
		NotImplementedError: schedule not in ['quad','linear','warmup10','warmup50','const','jsd','cosine']

	Returns:
		np.ndarray: (n_timestep)
	"""
	# beta_schedule function
	def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
		betas = linear_end * np.ones(n_timestep, dtype=np.float64)
		warmup_time = int(n_timestep * warmup_frac)
		betas[:warmup_time] = np.linspace(
			linear_start, linear_end, warmup_time, dtype=np.float64)
		return betas
	if schedule == 'quad':
		betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
							n_timestep, dtype=np.float64) ** 2
	elif schedule == 'linear':
		betas = np.linspace(linear_start, linear_end,
							n_timestep, dtype=np.float64)
	elif schedule == 'warmup10':
		betas = _warmup_beta(linear_start, linear_end,
							 n_timestep, 0.1)
	elif schedule == 'warmup50':
		betas = _warmup_beta(linear_start, linear_end,
							 n_timestep, 0.5)
	elif schedule == 'const':
		betas = linear_end * np.ones(n_timestep, dtype=np.float64)
	elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
		betas = 1. / np.linspace(n_timestep,
								 1, n_timestep, dtype=np.float64)
	elif schedule == "cosine":
		timesteps = (
			np.arange(n_timestep + 1, dtype=np.float64) /
			n_timestep + cosine_s
		)
		alphas = timesteps / (1 + cosine_s) * math.pi / 2
		alphas = np.power(np.cos(alphas),2)
		alphas = alphas / alphas[0]
		betas = 1 - alphas[1:] / alphas[:-1]
		betas = np.clip(betas, 0, 1-1e-3)
	else:
		raise NotImplementedError(schedule)
	return betas

class BaseNetwork(nn.Module):
	def __init__(self, init_type='kaiming', gain=0.02):
		super(BaseNetwork, self).__init__()
		self.init_type = init_type
		self.gain = gain

	def init_weights(self):
		"""
		initialize network's weights
		init_type: normal | xavier | kaiming | orthogonal
		https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
		"""
	
		def init_func(m:Union[nn.Module, torch.Tensor]):
			classname = m.__class__.__name__
			if classname.find('InstanceNorm2d') != -1:
				if hasattr(m, 'weight') and m.weight is not None:
					nn.init.constant_(m.weight.data, 1.0)
				if hasattr(m, 'bias') and m.bias is not None:
					nn.init.constant_(m.bias.data, 0.0)
			elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
				if self.init_type == 'normal':
					nn.init.normal_(m.weight.data, 0.0, self.gain)
				elif self.init_type == 'xavier':
					nn.init.xavier_normal_(m.weight.data, gain=self.gain)
				elif self.init_type == 'xavier_uniform':
					nn.init.xavier_uniform_(m.weight.data, gain=1.0)
				elif self.init_type == 'kaiming':
					nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
				elif self.init_type == 'orthogonal':
					nn.init.orthogonal_(m.weight.data, gain=self.gain)
				elif self.init_type == 'none':  # uses pytorch's default init method
					m.reset_parameters()
				else:
					raise NotImplementedError('initialization method [%s] is not implemented' % self.init_type)
				if hasattr(m, 'bias') and m.bias is not None:
					nn.init.constant_(m.bias.data, 0.0)

			self.apply(init_func)
			# propagate to children
			for m in self.children():
				if hasattr(m, 'init_weights'):
					m.init_weights(self.init_type, self.gain)

class Diffuser(BaseNetwork):
	def __init__(self, denoiser:Denoiser, beta_schedule:Dict, dpm_argv:Dict,  **kwargs):
		"""Diffuser

		Args:
			denoiser_paras (dict): hidden_dims:List[int], x_dim:int, depthmap_scale[Optional, float] = 256
			beta_schedule (dict): schedule:Literal['quad','linear','warmup10','warmup50','const','jsd','cosine'],
						n_timestep:int, linear_start:float=1e-6, linear_end:float=1e-2, cosine_s:float=8e-3
			loss_type (Literal['mse','mae']): str
			noise_std (float): amplitude of noise

		Raises:
			NotImplementedError: _description_
		"""
		super(Diffuser, self).__init__(**kwargs)
		self.x0_fn = denoiser
		self.beta_schedule = beta_schedule
		self.dpm_argv = dpm_argv

	@staticmethod
	def to_torch(x:Iterable[float], dtype=torch.float32, device=torch.device('cuda')):
		return torch.tensor(x, dtype=dtype, device=device)

	def set_loss(self, loss_fn):
		self.loss_fn = loss_fn

	def set_new_noise_schedule(self, device=torch.device('cuda')):
		to_torch = partial(self.to_torch, dtype=torch.float32, device=device)
		betas = make_beta_schedule(**self.beta_schedule)
		# betas = betas.detach().cpu().numpy() if isinstance(
		#     betas, torch.Tensor) else betas
		alphas = 1. - betas

		timesteps, = betas.shape
		self.num_timesteps = int(timesteps)
		
		gammas = np.cumprod(alphas, axis=0)
		gammas_prev = np.append(1., gammas[:-1])

		# calculations for diffusion q(x_t | x_{t-1}) and others
		self.register_buffer('gammas', to_torch(gammas))
		self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
		self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))
		
		# calculations for posterior q(x_{t-1} | x_t, x_0)
		posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
		# below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
		self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, np.spacing(np.float32(1))))))
		self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
		self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

	def predict_start_from_noise(self, x_t:torch.Tensor, t:torch.Tensor, noise:torch.Tensor):
		return (
			extract(self.sqrt_recip_gammas, t, x_t.shape) * x_t -
			extract(self.sqrt_recipm1_gammas, t, x_t.shape) * noise
		)

	def q_posterior(self, x_0_hat, x_t, t):
		posterior_mean = (
			extract(self.posterior_mean_coef1, t, x_t.shape) * x_0_hat +
			extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
		)
		posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
		return posterior_mean, posterior_log_variance_clipped

	def p_mean_variance(self, x_t:torch.Tensor, t:torch.Tensor, x_cond:torch.Tensor):
		# sample_gammas = extract(self.gammas, t, x_shape=(1, 1)).to(x_t.device)
		x_0_hat = self.x0_fn.forward(x_t, x_cond)  # x0 predictor
		model_mean, posterior_log_variance = self.q_posterior(
			x_0_hat=x_0_hat, x_t=x_t, t=t)
		return model_mean, posterior_log_variance

	def q_sample(self, x_0:torch.Tensor, sample_gammas:torch.Tensor, noise=None):
		noise = default(noise, torch.zeros_like(x_0))
		return (
			sample_gammas.sqrt() * x_0 +
			(1 - sample_gammas).sqrt() * noise
		)

	@torch.inference_mode()
	def p_sample(self, x_t, t, x_cond:torch.Tensor):
		model_mean, model_log_variance = self.p_mean_variance(
			x_t=x_t, t=t, x_cond=x_cond)
		noise = torch.zeros_like(x_t) if any(t>0) else torch.zeros_like(x_t)
		return model_mean + noise * (0.5 * model_log_variance).exp()

	@torch.inference_mode()
	def ddpm_sampling(self, x_T:torch.Tensor, x_cond:Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]) -> torch.Tensor:
		b = x_T.shape[0]
		x_t = x_T.clone()
		self.x0_fn.clear_buffer()
		self.x0_fn.restore_buffer(x_cond[:2])  # img, pcd
		for i in tqdm(reversed(range(0, self.num_timesteps)), desc='ddpm sampling', total=self.num_timesteps):
			t = torch.full((b,), i, device=x_t.device, dtype=torch.long)
			x_t = self.p_sample(x_t, t, x_cond=x_cond)  # img, pcd, init_Tcl, camera_info
		self.x0_fn.clear_buffer()
		return x_t
	

	@torch.inference_mode()
	def dpm_sampling(self, x_T:torch.Tensor, x_cond:Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]) -> torch.Tensor:
		def model_fn(x_t:torch.Tensor, t:torch.Tensor, x_cond:Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]):
			out = self.x0_fn.forward(x_t, x_cond)
			# If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
			# We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
			return out
		self.x0_fn.clear_buffer()
		self.x0_fn.restore_buffer(x_cond[:2])  # img, pcd, init_Tcl, camera_info
		noise_schedule = NoiseScheduleVP(schedule='discrete', alphas_cumprod=self.gammas)
		model_fn_continuous = model_wrapper(
			model_fn,
			noise_schedule,
			model_kwargs={"x_cond":x_cond},
			model_type='x_start',
			guidance_type='uncond'
		)
		dpm_solver = DPM_Solver(
			model_fn_continuous,
			noise_schedule,
			algorithm_type="dpmsolver++"
		)
		x_0_hat = dpm_solver.sample(
			x_T,
			**self.dpm_argv
		)
		self.x0_fn.clear_buffer()
		return x_0_hat
	
	@torch.no_grad()
	def dpm_sampling_with_guidance(self, x_T:torch.Tensor, x_cond:Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict], corr_data:Dict[str,np.ndarray], classifier_fn_argv:Dict, guidance_scale:float, classifier_t_threshold:float) -> torch.Tensor:
		# x_cond: img, pcd, Tcl, camera_info
		def model_fn(x_t:torch.Tensor, t:torch.Tensor, x_cond:Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]):
			out = self.x0_fn.forward(x_t, x_cond)
			# If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
			# We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
			return out
		def classifier_fn_wrapper(x_t:torch.Tensor, t:torch.Tensor, condition:torch.Tensor, init_gt:torch.Tensor, camera_info:Dict):
			loss = classifer_guidance.classifer_fn(x_t, init_gt, camera_info)
			return loss
		self.x0_fn.clear_buffer()
		self.x0_fn.restore_buffer(x_cond[:2])  # img, pcd, init_Tcl, camera_info
		noise_schedule = NoiseScheduleVP(schedule='discrete', alphas_cumprod=self.gammas)
		classifer_guidance = GuidanceSampler(**classifier_fn_argv, corr_data=corr_data)
		model_fn_continuous = model_wrapper(
			model_fn,
			noise_schedule,
			model_kwargs={"x_cond":x_cond},
			model_type='x_start',
			guidance_type='classifier',
			guidance_scale=guidance_scale,
			classifier_fn=classifier_fn_wrapper,
			classifier_kwargs=dict(init_gt=x_cond[-2], camera_info=x_cond[-1]),
			classifier_t_threshold=classifier_t_threshold
		)
		dpm_solver = DPM_Solver(
			model_fn_continuous,
			noise_schedule,
			algorithm_type="dpmsolver++"
		)
		x_0_hat = dpm_solver.sample(
			x_T,
			**self.dpm_argv
		)
		self.x0_fn.clear_buffer()
		return x_0_hat

	def forward(self, x_0:torch.Tensor, x_cond:Tuple[torch.Tensor, torch.Tensor, Dict], noise=None) -> torch.Tensor:
		"""Training `theta`(x_t, t) = `epsilon_0`

		Args:
			x_0 (torch.Tensor): gt variable
			x_cond (Tuple[torch.Tensor]): condition
			noise (_type_, optional): added Guassian Noise. Defaults to None.

		Returns:
			loss: error between x0_hat and x0
		"""
		b = x_0.shape[0]
		t = torch.randint(1, self.num_timesteps, (b,), device=x_0.device).long()  # (b,)
		gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))  # (b, 1, 1)
		sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))  # (b, 1, 1)
		sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=x_0.device) + gamma_t1  # randomly linear interpolation between gamma1 and gamm2
		noise = default(noise, torch.zeros_like(x_0))  # x_T = 0
		x_t = self.q_sample(
			x_0=x_0, sample_gammas=sample_gammas.view(b,1), noise=noise)
		x_0_hat = self.x0_fn.forward(x_t, x_cond)  # img, pcd, init_Tcl, camera_info
		loss = self.loss_fn(x_0_hat, x_0)
		return loss, x_0_hat
	
class GuidanceSampler:
	def __init__(self, loss_fn:Callable, corr_data:Dict, max_corr_dist:float, proj_constraint:bool=True):
		corr_data.update({"max_dist":max_corr_dist, "proj_constraint":proj_constraint})
		self.corr_data = corr_data
		self.loss = loss_fn

	def classifer_fn(self, se3_x:torch.Tensor, init_gt:torch.Tensor, camera_info:Dict):
		delta_se3 = se3.exp(se3_x)
		se3_extran = delta_se3 @ init_gt # (1,4,4)
		Tcl = se3_extran.squeeze(0).cpu().detach().numpy()
		assert np.ndim(Tcl) == 2, 'classifer guidances can only handle batch = 1, get {}'.format(Tcl.shape[0])
		corr_res_list = CBABatchCorr(**self.corr_data, Tcl=Tcl)
		if len(corr_res_list) == 0:
			return 0
		loss = 0
		for corr_res in corr_res_list:
			relpose = torch.from_numpy(corr_res['relpose']).to(se3_x).unsqueeze(0)  # (1, 4, 4)
			src_pcd = torch.from_numpy(corr_res['src_pcd']).to(se3_x).transpose(-1,-2).unsqueeze(0)  # (1, 3, N)
			src_pcd = se3.transform(se3_extran, src_pcd)
			tgt_pcd = se3.transform(relpose, src_pcd)
			tgt_proj = project_pc2image(tgt_pcd, camera_info)  # (1, 2, N)
			tgt_kpt = torch.from_numpy(corr_res['tgt_kpt']).to(se3_x).transpose(-1,-2).unsqueeze(0) # (1, 2, N)
			frame_loss = self.loss(tgt_proj, tgt_kpt)
			loss += frame_loss
		loss /= len(corr_res_list)
		return -loss

	@staticmethod
	def check_data(corr_data:Dict):
		# dict(src_pcd=pcd, src_kpt=self.kpts[index], tgt_kpt_list=tgt_kpt_list, match_list=match_list, scale=self.scale)
		return True
