import argparse
import dataclasses
from typing import List, Protocol, Union, Optional, Type, TypeVar
import typing
import omegaconf
import logging
import torch
import numpy as np
import random
import math
from contextlib import contextmanager
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from torch.optim import Optimizer
import wandb
import torch.distributed


class ConfigProtocolWithWandb(Protocol):
	wandb_project: str | None
	__dataclass_fields__: dict


class ConfigProtocol(Protocol):
	__dataclass_fields__: dict


T = TypeVar('T', bound=ConfigProtocol)

def parse_args_into_config(config_class: Type[T], logger: logging.Logger) -> T | None:
	bool_arg = lambda x: (str(x).lower() in ['true', '1', 't', 'y', 'yes'])  # noqa: E731
	parser = argparse.ArgumentParser(description='Trainer')

	# Add all config options to the parser
	for field in dataclasses.fields(config_class):
		field_type = get_inner_type_if_optional(field.type)
		field_name = field.name.replace('_', '-')

		if field_type == List[str]:
			parser.add_argument(f'--{field_name}', type=str, action='append')
		elif field_type == bool:
			parser.add_argument(f'--{field_name}', type=bool_arg)
		else:
			parser.add_argument(f'--{field_name}', type=field_type)
	
	try:
		args = parser.parse_args()
		config = omegaconf.OmegaConf.structured(config_class)

		for field in dataclasses.fields(config_class):
			arg_val = getattr(args, field.name, None)
			if arg_val is not None:
				omegaconf.OmegaConf.update(config, field.name, arg_val, merge=True)
		
		# Check if any fields are missing
		for field in dataclasses.fields(config_class):
			if omegaconf.OmegaConf.is_missing(config, field.name):
				raise ValueError(f"Missing required config value for {field.name}")
	except Exception as e:
		logger.error(f"Error parsing args: {e}")
		return None
	
	return config


def get_inner_type_if_optional(ty):
	if typing.get_origin(ty) is Union and type(None) in typing.get_args(ty):
		return ty.__args__[0]
	return ty


@contextmanager
def temprngstate(new_seed: Optional[int] = None):
	"""
	Context manager that saves and restores the RNG state of PyTorch, NumPy and Python.
	If new_seed is not None, the RNG state is set to this value before the context is entered.
	"""

	# Save RNG state
	old_torch_rng_state = torch.get_rng_state()
	old_torch_cuda_rng_state = torch.cuda.get_rng_state()
	old_numpy_rng_state = np.random.get_state()
	old_python_rng_state = random.getstate()

	# Set new seed
	if new_seed is not None:
		torch.manual_seed(new_seed)
		torch.cuda.manual_seed(new_seed)
		np.random.seed(new_seed)
		random.seed(new_seed)

	yield

	# Restore RNG state
	torch.set_rng_state(old_torch_rng_state)
	torch.cuda.set_rng_state(old_torch_cuda_rng_state)
	np.random.set_state(old_numpy_rng_state)
	random.setstate(old_python_rng_state)


def _get_cosine_schedule_with_warmup_lr_lambda(
	current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_ratio: float
):
	if current_step < num_warmup_steps:
		return float(current_step) / float(max(1, num_warmup_steps))
	
	progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
	r = 1.0 - min_lr_ratio
	return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * r + min_lr_ratio


def get_cosine_schedule_with_warmup(
	optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, min_lr_ratio: float = 0.0
):
	lr_lambda = partial(
		_get_cosine_schedule_with_warmup_lr_lambda,
		num_warmup_steps=num_warmup_steps,
		num_training_steps=num_training_steps,
		num_cycles=num_cycles,
		min_lr_ratio=min_lr_ratio,
	)
	return LambdaLR(optimizer, lr_lambda, last_epoch)


def main(config_cls: Type[ConfigProtocolWithWandb], trainer_cls: Type, logger_name: str):
	# Logging
	logger = logging.getLogger(logger_name)
	logging.basicConfig(format='%(asctime)s [%(name)s] [%(levelname)s] [%(funcName)s] - %(message)s')
	logger.setLevel(logging.INFO)

	# Parse args
	config = parse_args_into_config(config_cls, logger)
	if config is None:
		return
	
	wc = omegaconf.OmegaConf.to_container(config, resolve=True)
	assert isinstance(wc, dict)
	w = wandb.init(config=wc, project=config.wandb_project)
	assert w is not None
	with w:
		assert wandb.run is not None

		trainer = trainer_cls(config=config, run_id=wandb.run.id, logger=logger)
		trainer.train()


class MetricCounters:
	true_positives: torch.Tensor
	false_positives: torch.Tensor
	true_negatives: torch.Tensor
	false_negatives: torch.Tensor

	def __init__(self, num_classes: int, device: Union[torch.device, str]):
		self.true_positives = torch.zeros(num_classes, device=device, requires_grad=False, dtype=torch.int32)
		self.false_positives = torch.zeros(num_classes, device=device, requires_grad=False, dtype=torch.int32)
		self.true_negatives = torch.zeros(num_classes, device=device, requires_grad=False, dtype=torch.int32)
		self.false_negatives = torch.zeros(num_classes, device=device, requires_grad=False, dtype=torch.int32)
	
	def accuracy(self, eps: float = 1e-8) -> torch.Tensor:
		"""
		Compute the accuracy.
		Args:
			eps: Epsilon to avoid division by zero
		Returns:
			[num_classes] tensor of mean accuracies per class
		"""
		denom = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
	
		return (self.true_positives + self.true_negatives) / (denom.float() + eps)
	
	def precision(self, eps: float = 1e-8) -> torch.Tensor:
		"""
		Compute the precision.
		Args:
			eps: Epsilon to avoid division by zero
		Returns:
			[num_classes] tensor of mean precisions per class
		"""
		denom = self.true_positives + self.false_positives
		
		return self.true_positives / (denom.float() + eps)

	def recall(self, eps: float = 1e-8) -> torch.Tensor:
		"""
		Compute the recall.
		Args:
			eps: Epsilon to avoid division by zero
		Returns:
			[num_classes] tensor of mean recalls per class
		"""
		denom = self.true_positives + self.false_negatives
		
		return self.true_positives / (denom.float() + eps)

	def f1(self, eps: float = 1e-8) -> torch.Tensor:
		"""
		Compute the F1 score.
		Args:
			eps: Epsilon to avoid division by zero
		Returns:
			[num_classes] tensor of mean F1 scores per class
		"""
		precision = self.precision(eps=eps)
		recall = self.recall(eps=eps)
		return 2 * precision * recall / (precision + recall + eps)


def distributed_rank():
	if not torch.distributed.is_initialized():
		return 0
	
	return torch.distributed.get_rank()


def distributed_world_size():
	if not torch.distributed.is_initialized():
		return 1
	
	return torch.distributed.get_world_size()


def distributed_setup():
	torch.distributed.init_process_group(backend="nccl", init_method="env://")


def distributed_cleanup():
	torch.distributed.destroy_process_group()


def print_rank_0(*args, **kwargs):
	if distributed_rank() == 0:
		print(*args, **kwargs)


def log_rank_0(logger: logging.Logger, level: int, *args, **kwargs):
	if distributed_rank() == 0:
		logger.log(level, *args, **kwargs)