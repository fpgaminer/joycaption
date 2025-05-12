#!/usr/bin/env python3
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, LlavaForConditionalGeneration
from dataclasses import dataclass
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Optional
import torch.backends.cudnn
import torch.backends.cuda
import math
from PIL import Image
from transformers import get_scheduler
from tqdm import tqdm
import numpy as np
from tqdm.contrib.logging import logging_redirect_tqdm
import random
import torch.amp
import wandb
import omegaconf
from utils import parse_args_into_config, get_cosine_schedule_with_warmup, temprngstate, distributed_rank, distributed_setup, distributed_cleanup, distributed_world_size, log_rank_0
from torch.distributed.elastic.multiprocessing.errors import record
from torch.optim.optimizer import Optimizer
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
import itertools
from peft import LoraConfig, get_peft_model, TaskType
import json


DTYPE_MAP = { 'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16 }


@dataclass
class Config:
	output_dir: Path = Path("checkpoints")               # Output directory
	wandb_project: Optional[str] = None                  # Wandb project
	device_batch_size: int = 1                           # Device batch size
	batch_size: int = 32                                 # Actual batch size; gradient accumulation is used on device_batch_size to achieve this
	learning_rate: float = 5e-5                          # Learning rate

	warmup_samples: int = 0                              # Warmup samples
	max_samples: int = 400000                            # Max samples trained for in this session
	save_every: int = 50000                              # Save a checkpoint every n samples (approx)
	test_every: int = 50000                              # Test every n samples (approx)
	grad_scaler: bool = False                            # Use gradient scaler
	lr_scheduler_type: str = "cosine"                    # Learning rate scheduler type
	min_lr_ratio: float = 0.0                            # Minimum learning rate ratio for scheduler
	allow_tf32: bool = True                              # Allow tf32
	seed: int = 42                                       # Random seed
	num_workers: int = 2                                 # Num workers

	optimizer_type: str = "adamw"                        # Optimizer type
	adam_beta1: float = 0.9                              # Adam beta1
	adam_beta2: float = 0.999                            # Adam beta2
	adam_eps: float = 1e-8                               # Adam epsilon
	adam_weight_decay: float = 0.00                      # Adam weight decay

	clip_grad_norm: Optional[float] = 1.0                # Clip gradient norm

	dataset: str = "your_dataset.json"                   # Dataset path (parquet)
	images_path: Path = Path("../data/resized-384-squish")   # Images path
	finetune: str = "fancyfeast/llama-joycaption-beta-one-hf-llava"   # Model to finetune from
	gradient_checkpointing: bool = True                  # Use gradient checkpointing
	test_size: int = 128                                 # Test size
	grad_scaler_init: float = 2**16                      # Initial grad scaler

	text_model_dtype: str = "bfloat16"                   # Text model dtype
	pre_test: bool = True                                # Pre-test the model
	lora_r: int = 64                                     # LORA rank
	lora_alpha: int = 64                                 # LORA alpha
	lora_dropout: float = 0.0                            # LORA dropout


@record
def main():
	# Logging
	logger = logging.getLogger(f'Process-{distributed_rank()}')
	logging.basicConfig(format='%(asctime)s [%(name)s] [%(levelname)s] [%(funcName)s] - %(message)s')
	logger.setLevel(logging.INFO)

	if distributed_rank() == 0:
		# Parse args
		config = parse_args_into_config(Config, logger)
		if config is None:
			torch.distributed.broadcast_object_list([None, None])
			return
		
		# Start
		wc = omegaconf.OmegaConf.to_container(config, resolve=True)
		assert isinstance(wc, dict)
		w = wandb.init(config=wc, project=config.wandb_project)
		assert w is not None
		with w:
			assert wandb.run is not None

			if wandb.run.resumed and config.resume is None:
				raise NotImplementedError("Resuming from a checkpoint is not yet supported")
			
			# Broadcast the config and run_id to all other processes
			torch.distributed.broadcast_object_list([config, wandb.run.id])

			logger.info("Rank 0 starting training...")
			trainer = MainTrainer(config=config, run_id=wandb.run.id, logger=logger)
			trainer.train()
	else:
		objects = [None, None]
		logger.info(f"Rank {distributed_rank()} waiting for config...")
		torch.distributed.broadcast_object_list(objects)
		config, run_id = objects

		if config is None or run_id is None:
			logger.info(f"Rank {distributed_rank()} exiting...")
			return
		
		logger.info(f"Rank {distributed_rank()} starting training...")
		trainer = MainTrainer(config=config, run_id=run_id, logger=logger)
		trainer.train()


class MainTrainer:
	config: Config
	run_id: str
	rank: int
	logger: logging.Logger
	model: nn.Module
	text_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast

	optimizer: Optimizer
	device: str
	device_batch_size: int
	gradient_accumulation_steps: int
	test_every_step: int
	save_every_step: int
	total_steps: int
	total_device_batches: int
	world_size: int

	def __init__(self, config: Config, run_id: str, logger: logging.Logger):
		self.config = config
		self.rank = distributed_rank()
		self.run_id = run_id
		self.logger = logger
		self.device = f"cuda:{torch.cuda.current_device()}"
		self.world_size = distributed_world_size()

		self.config.output_dir = Path(config.output_dir)

		if config.allow_tf32:
			torch.backends.cuda.matmul.allow_tf32 = True
			torch.backends.cudnn.allow_tf32 = True

		self.device_batch_size = min(config.batch_size // self.world_size, config.device_batch_size)
		self.gradient_accumulation_steps = config.batch_size // (self.device_batch_size * self.world_size)
		self.test_every_step = int(math.ceil(config.test_every / config.batch_size))
		self.save_every_step = int(math.ceil(config.save_every / config.batch_size))
		self.total_steps = self.config.max_samples // self.config.batch_size
		self.total_device_batches = self.total_steps * self.gradient_accumulation_steps

		assert config.batch_size == self.device_batch_size * self.gradient_accumulation_steps * self.world_size, "Batch size must be a multiple of device batch size"
	
	def build_model(self):
		# Text Model
		self.logger.info("Building text model...")

		tokenizer = AutoTokenizer.from_pretrained(self.config.finetune, use_fast=True)
		assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Expected PreTrainedTokenizer, got {type(tokenizer)}"

		model = LlavaForConditionalGeneration.from_pretrained(self.config.finetune, device_map=self.rank, torch_dtype="bfloat16")

		# Enable gradient checkpointing
		if self.config.gradient_checkpointing:
			model.gradient_checkpointing_enable()

		self.text_tokenizer = tokenizer

		# LORA
		lora_config = LoraConfig(
			task_type=TaskType.CAUSAL_LM,
			inference_mode=False,
			r=self.config.lora_r,
			lora_alpha=self.config.lora_alpha,
			lora_dropout=self.config.lora_dropout,
			target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
			bias="none",
		)
		self.model = get_peft_model(model, lora_config)
		self.model.print_trainable_parameters()
		total_params = sum(p.numel() for p in self.model.parameters())
		total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		self.logger.info(f"Total LLM parameters: {total_params:,}")
		self.logger.info(f"Total LLM trainable parameters: {total_trainable_params:,}")
		
		self.model_module = self.model

		# Distributed training
		if self.world_size > 1:
			self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank], output_device=self.rank, gradient_as_bucket_view=True, find_unused_parameters=True)

	def build_dataset(self):
		self.logger.info("Building dataset...")

		# Load dataset
		source_ds = json.loads(Path(self.config.dataset).read_text())
		assert isinstance(source_ds, list)

		# Strip <image>
		for i in range(len(source_ds)):
			for j in range(len(source_ds[i]['messages'])):
				source_ds[i]['messages'][j]['content'] = source_ds[i]['messages'][j]['content'].replace("<image>", "").strip()
		
		# Preprocess all images
		for example in tqdm(source_ds, desc="Preprocessing images", dynamic_ncols=True, disable=self.rank != 0):
			assert len(example['images']) == 1
			image = Image.open(self.config.images_path / example['images'][0])
			if image.size != (384, 384):
				image = image.resize((384, 384), Image.LANCZOS) # type: ignore
			image = image.convert("RGB")
			pixel_values = TVF.pil_to_tensor(image)
			example['pixel_values'] = pixel_values
		
		# Shuffle and split
		rng = random.Random(self.config.seed)
		rng.shuffle(source_ds)
		test_examples = source_ds[:self.config.test_size]
		train_examples = source_ds[self.config.test_size:]

		self.train_dataset = ImageDataset(
			examples=train_examples,
			tokenizer=self.text_tokenizer,
			image_token_id=self.model.config.image_token_index,
			image_seq_length=self.model.config.image_seq_length,
		)

		self.test_dataset = ImageDataset(
			examples=test_examples,
			tokenizer=self.text_tokenizer,
			image_token_id=self.model.config.image_token_index,
			image_seq_length=self.model.config.image_seq_length,
		)
	
	def build_dataloader(self):
		self.logger.info("Building dataloader...")
		
		self.train_sampler = BetterDistributedSampler(
			self.train_dataset,
			num_replicas=self.world_size,
			rank=self.rank,
			shuffle=True,
			drop_last=True,
			seed=self.config.seed
		)

		self.test_sampler = BetterDistributedSampler(
			self.test_dataset,
			num_replicas=self.world_size,
			rank=self.rank,
			shuffle=False,
			drop_last=False,
			seed=self.config.seed
		)

		self.train_dataloader = DataLoader(
			self.train_dataset,
			batch_size=self.device_batch_size,
			sampler=self.train_sampler,
			num_workers=self.config.num_workers,
			pin_memory=True,
			drop_last=True,
			pin_memory_device=self.device,
			collate_fn=self.train_dataset.collate_fn,
		)

		self.test_dataloader = DataLoader(
			self.test_dataset,
			batch_size=self.device_batch_size,
			sampler=self.test_sampler,
			num_workers=self.config.num_workers,
			pin_memory=True,
			drop_last=False,
			pin_memory_device=self.device,
			collate_fn=self.test_dataset.collate_fn,
		)
	
	def build_optimizer(self):
		self.logger.info("Building optimizer...")
		self.optimized_params = [{
			'params': [p for p in self.model.parameters() if p.requires_grad],
		}]

		if self.config.optimizer_type == "adamw":
			optimizer_cls = torch.optim.AdamW
			kwargs = {
				'lr': self.config.learning_rate,
				'betas': (self.config.adam_beta1, self.config.adam_beta2),
				'eps': self.config.adam_eps,
				'weight_decay': self.config.adam_weight_decay,
			}
		else:
			raise ValueError(f"Unknown optimizer type {self.config.optimizer_type}")
		
		self.optimizer = optimizer_cls(self.optimized_params, **kwargs)
		self.optimized_params = list(itertools.chain(*[group['params'] for group in self.optimized_params]))
	
	def build_lr_scheduler(self):
		self.logger.info("Building lr scheduler...")
		num_warmup_steps = int(math.ceil(self.config.warmup_samples / self.config.batch_size))

		if self.config.lr_scheduler_type == "cosine":
			self.lr_scheduler = get_cosine_schedule_with_warmup(
				optimizer=self.optimizer,
				num_warmup_steps=num_warmup_steps,
				num_training_steps=self.total_steps,
				min_lr_ratio=self.config.min_lr_ratio,
			)
		elif self.config.lr_scheduler_type == "1cycle":
			self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
				optimizer=self.optimizer,
				max_lr=self.config.learning_rate,
				total_steps=self.total_steps,
			)
		elif self.config.lr_scheduler_type == "lr_finder":
			self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
				optimizer=self.optimizer,
				lr_lambda=lambda step: np.geomspace(1e-7, 10, self.total_steps)[step - 1],
			)
		else:
			self.lr_scheduler = get_scheduler(self.config.lr_scheduler_type, self.optimizer, num_warmup_steps, self.total_steps)
	
	def train(self):
		# Seed
		seed = hash((self.config.seed, self.rank)) & 0xffffffff   # NumPy requires 32-bit seeds
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)

		self.scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=self.config.grad_scaler, init_scale=self.config.grad_scaler_init, growth_interval=500)
		self.build_model()
		self.build_dataset()
		self.build_dataloader()
		self.build_optimizer()
		self.build_lr_scheduler()

		device_step = 0

		# Wandb
		if self.rank == 0 and self.config.wandb_project is not None:
			wandb.watch(self.model)
		
		# Pre-test
		self.global_step = 0
		self.global_samples_seen = 0

		if self.config.pre_test:
			self.do_validation('test', self.test_dataloader, self.test_dataset)

		self.logger.info("Starting training...")
		loss_sum = torch.tensor(0.0, device=self.device, requires_grad=False, dtype=torch.float32)
		dataloader_iter = iter(self.train_dataloader)

		pbar = tqdm(total=self.total_device_batches * self.device_batch_size * self.world_size, initial=device_step * self.device_batch_size * self.world_size, dynamic_ncols=True, smoothing=0.01, disable=self.rank != 0)
		with logging_redirect_tqdm():
			for device_step in range(device_step, self.total_device_batches):
				self.global_step = device_step // self.gradient_accumulation_steps
				self.global_samples_seen = (device_step + 1) * self.device_batch_size * self.world_size

				self.model.train()

				# Get batch
				try:
					batch = next(dataloader_iter)
				except StopIteration:
					logging.warning("Dataloader iterator exhausted. Resetting...")
					self.train_sampler.set_epoch(self.train_sampler.epoch + 1)  # This is important to ensure the data is re-shuffled after every use
					dataloader_iter = iter(self.train_dataloader)
					batch = next(dataloader_iter)
				
				is_last_device_step = (device_step + 1) % self.gradient_accumulation_steps == 0
				is_last_step = (self.global_step + 1) == self.total_steps

				# Forward pass
				loss, _ = self.run_model(batch)
				loss = loss / self.gradient_accumulation_steps
				loss_sum.add_(loss.detach())

				if torch.isnan(loss) or torch.isinf(loss):
					self.logger.error(f"Loss is NaN or Inf: {loss}")
					raise RuntimeError("Loss is NaN or Inf")
				
				# Backward pass
				self.scaler.scale(loss).backward() # type: ignore

				# Take a step if accumulation is done
				if is_last_device_step:
					# Reduce loss_sum across devices for logging
					torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)

					# Unscale the gradients before clipping
					self.scaler.unscale_(self.optimizer)

					# Clip gradients
					if self.config.clip_grad_norm is not None:
						torch.nn.utils.clip_grad.clip_grad_norm_(self.optimized_params, self.config.clip_grad_norm)
					
					# Take a step
					self.scaler.step(self.optimizer)
					self.scaler.update()
					self.lr_scheduler.step()
					self.optimizer.zero_grad(set_to_none=True)

					if self.rank == 0:
						wandb.log({
							"train/loss": (loss_sum.item() / self.world_size),
							"train/lr": self.lr_scheduler.get_last_lr()[0],
							"train/samples": self.global_samples_seen,
							"train/scaler": self.scaler.get_scale(),
						}, step=self.global_step)

					loss_sum.zero_()

					# Save checkpoint
					# Saved every save_every steps and at the end of training
					if self.save_every_step > 0 and ((self.global_step + 1) % self.save_every_step == 0 or is_last_step):
						self.save_checkpoint()

					# Validation
					# Run every test_every steps and at the end of training
					if self.test_every_step > 0 and ((self.global_step + 1) % self.test_every_step == 0 or is_last_step):
						self.do_validation('test', self.test_dataloader, self.test_dataset)
				
				pbar.update(self.device_batch_size * self.world_size)
			
			pbar.close()
	
	def save_checkpoint(self):
		log_rank_0(self.logger, logging.INFO, "Saving checkpoint...")

		#sampler_epoch = self.train_sampler.epoch
		sampler_index = self.global_samples_seen // self.world_size  # NOTE: sampler_index is in terms of "samples", not batches or steps
		sampler_index = sampler_index % (len(self.train_dataloader) * self.device_batch_size)

		base_path = self.config.output_dir / self.run_id
		path = base_path / f"samples_{self.global_samples_seen}"
		tmp_path = base_path / "tmp"
		tmp_path.mkdir(parents=True, exist_ok=True)

		if self.rank == 0:
			self.model.save_pretrained(tmp_path / "model")

		# Synchonize, so that all ranks are done before we move the checkpoint into place
		if self.world_size > 1:
			torch.distributed.barrier()
		
		# Move checkpoint into place
		if self.rank == 0:
			tmp_path.rename(path)
	
	def run_model(self, batch: dict, reduction: str = 'mean') -> tuple[torch.Tensor, torch.Tensor]:
		# Move to device
		pixel_values = batch['pixel_values'].to(self.device, non_blocking=True)
		input_ids = batch['input_ids'].to(self.device, non_blocking=True)
		attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
		labels = batch['labels'].to(self.device, non_blocking=True)[:, 1:]

		# Normalize the image
		pixel_values = pixel_values / 255.0
		pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
		pixel_values = pixel_values.to(torch.bfloat16)

		# Forard the model
		text_outputs = self.model(
			input_ids=input_ids[:, :-1],  # Skip the last token, wasted compute
			pixel_values=pixel_values,
			attention_mask=attention_mask[:, :-1],
			use_cache=False,
		)

		# Compute loss
		# We only compute loss on the labelled section of the logits
		logits = text_outputs.logits
		assert logits.shape == (labels.shape[0], labels.shape[1], self.model.vocab_size), f"Expected {labels.shape[0], labels.shape[1], self.model.vocab_size}, got {logits.shape}"
		logits = logits.reshape(-1, logits.shape[-1])  # Flatten to B*len x 1024
		labels = labels.reshape(-1)           # Flatten to B*len
		loss = F.cross_entropy(logits, labels, reduction=reduction)  # B*len losses (possibly reduced)

		if reduction == 'none':
			loss = loss.reshape(labels.shape[0], labels.shape[1])  # Reshape back to B x len
		
		if self.rank == 0 and not hasattr(self, 'debug_batch_saved'):
			torch.save({
				'input_ids': input_ids,
				'labels': labels,
				'attention_mask': attention_mask,
				'pixel_values': pixel_values,
				'text_outputs': text_outputs,
			}, 'debug_batch.pt')
			self.debug_batch_saved = True

		return loss, text_outputs.logits
	
	@torch.no_grad()
	def do_validation(self, metric_name: str, dataloader: DataLoader, dataset: "ImageDataset"):
		log_rank_0(self.logger, logging.INFO, f"Running {metric_name}...")

		with temprngstate(42):
			self.model.eval()

			dataloader_iter = iter(dataloader)
			n_items = len(dataset)
			loss_sum = torch.tensor(0.0, device=self.device, requires_grad=False, dtype=torch.float32)

			pbar = tqdm(total=n_items, dynamic_ncols=True, desc=metric_name, disable=self.rank != 0)
			for batch in dataloader_iter:
				# Forward pass
				loss, _ = self.run_model(batch, reduction='mean')
				loss_sum.add_(loss.detach())

				pbar.update(self.device_batch_size * self.world_size)
			
			pbar.close()

			torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
			loss_sum = loss_sum / len(dataloader)
			loss_sum = loss_sum / self.world_size

			# All other ranks are done
			if self.rank != 0:
				return
			
			results = {
				f"{metric_name}/samples": self.global_samples_seen,
				f"{metric_name}/loss": loss_sum.item(),
			}
			wandb.log(results, step=self.global_step)


class ImageDataset(Dataset):
	def __init__(self, examples: list, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, image_token_id: int, image_seq_length: int):
		self.examples = examples
		self.tokenizer = tokenizer
		self.image_token_id = image_token_id
		self.image_seq_length = image_seq_length
		self.pad_token_id = tokenizer.pad_token_id

	def __len__(self):
		return len(self.examples)
	
	def __getitem__(self, idx: int):
		pixel_values = self.examples[idx]['pixel_values']
		messages = self.examples[idx]['messages']

		# Build the conversation
		convo = [
			{
				"role": "system",
				"content": "You are a helpful image captioner.",
			}
		]

		convo.extend(messages)

		# Format the conversation
		convo_string = self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False)
		assert isinstance(convo_string, str)

		# Tokenize the conversation
		convo_tokens = self.tokenizer.encode(convo_string, add_special_tokens=False, truncation=False)

		# Repeat the image tokens
		input_tokens = []
		for token in convo_tokens:
			if token == self.image_token_id:
				input_tokens.extend([self.image_token_id] * self.image_seq_length)
			else:
				input_tokens.append(token)
		
		input_ids = torch.tensor(input_tokens, dtype=torch.long)
		labels = torch.tensor(input_tokens, dtype=torch.long)
		attention_mask = torch.ones_like(input_ids)

		# Mask out everything but the assistant's response
		# WARNING: Assumes the assistant's response is the last message
		eot_id_indices = (input_ids == self.tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[0].tolist()
		assert len(eot_id_indices) == 3, f"Expected 3 <|eot_id|> tokens, got {len(eot_id_indices)}"

		end_header_indices = (input_ids == self.tokenizer.convert_tokens_to_ids("<|end_header_id|>")).nonzero(as_tuple=True)[0].tolist()
		assert len(end_header_indices) == 3, f"Expected 3 <|end_header_id|> tokens, got {len(end_header_indices)}"

		label_start_idx = end_header_indices[2] + 1   # Start of the labels (inclusive) (everything after assistant<|end_header_id|>)
		labels[:label_start_idx] = -100

		return {
			'pixel_values': pixel_values,
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'labels': labels,
		}
	
	def collate_fn(self, batch: list[dict]) -> dict:
		# Filter out images that failed to load
		batch = [item for item in batch if item['pixel_values'] is not None]

		# Pad input_ids and attention_mask
		max_length = max(item['input_ids'].shape[0] for item in batch)
		n_pad = [max_length - item['input_ids'].shape[0] for item in batch]
		input_ids = torch.stack([torch.nn.functional.pad(item['input_ids'], (0, n), value=self.pad_token_id) for item, n in zip(batch, n_pad)])
		attention_mask = torch.stack([torch.nn.functional.pad(item['attention_mask'], (0, n), value=0) for item, n in zip(batch, n_pad)])
		labels = torch.stack([torch.nn.functional.pad(item['labels'], (0, n), value=-100) for item, n in zip(batch, n_pad)])

		# Stack pixel values
		pixel_values = torch.stack([item['pixel_values'] for item in batch])

		return {
			'pixel_values': pixel_values,
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'labels': labels,
		}


class BetterDistributedSampler(DistributedSampler):
	def __init__(
		self,
		dataset: Dataset,
		num_replicas: Optional[int] = None,
		rank: Optional[int] = None,
		shuffle: bool = True,
		seed: int = 0,
		drop_last: bool = False,
	) -> None:
		super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
		self.resume_index = None
	
	def set_state(self, epoch: int, index: int) -> None:
		"""
		Sets the epoch and fast forwards the iterator to the given index.
		Needs to be called before the dataloader is iterated over.
		"""
		self.set_epoch(epoch)
		self.resume_index = index

	def __iter__(self):
		i = super().__iter__()

		if self.resume_index is not None:
			for _ in range(self.resume_index):
				next(i)
			self.resume_index = None
		
		return i


if __name__ == "__main__":
	distributed_setup()
	torch.cuda.set_device(distributed_rank())
	main()
	distributed_cleanup()
