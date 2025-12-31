#!/usr/bin/env python3
"""
Use JoyCaption to caption images.
"""
import argparse
import dataclasses
import glob as glob_module
import json
import logging
import os
import random
from pathlib import Path, PurePath

import PIL.Image
import torch
import torch.amp
import torchvision.transforms.functional as TVF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
	AutoTokenizer,
	LlavaForConditionalGeneration,
	PreTrainedTokenizer,
	PreTrainedTokenizerFast,
)

def none_or_type(value, desired_type):
	if value == "None":
		return None
	return desired_type(value)

parser = argparse.ArgumentParser()
parser.add_argument("--glob", type=str, help="Glob pattern to find images")
parser.add_argument("--filelist", type=str, help="File containing list of images")
parser.add_argument("--prompt", type=str, help="Prompt to use")
parser.add_argument("--prompt-file", type=str, help="JSON file containing prompts to use")
parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling")
parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
parser.add_argument("--top-p", type=lambda x: none_or_type(x, float), default=0.9, help="Top-p sampling")
parser.add_argument("--top-k", type=lambda x: none_or_type(x, int), default=None, help="Top-k sampling")
parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum length of the generated caption (in tokens)")
parser.add_argument("--num-workers", type=int, default=4, help="Number of workers loading images in parallel")
parser.add_argument("--model", type=str, default="fancyfeast/llama-joycaption-beta-one-hf-llava", help="Model to use")
parser.add_argument("--prepend", type=str, default="", help="String to prepend to all captions")
parser.add_argument("--append", type=str, default="", help="String to append to all captions")


PIL.Image.MAX_IMAGE_PIXELS = 933120000   # Quiets Pillow from giving warnings on really large images (WARNING: Exposes a risk of DoS from malicious images)


@dataclasses.dataclass
class Prompt:
	prompt: str
	weight: float


@torch.no_grad()
def main():
	# Logging
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

	# Parse arguments
	args = parser.parse_args()
	logging.info(f"Arguments: {args}")

	# Make sure we have a prompt or a prompt file
	prompts = parse_prompts(args.prompt, args.prompt_file)

	# Find the images
	image_paths = find_images(args.glob, args.filelist)
	if len(image_paths) == 0:
		logging.warning("No images found")
		return
	logging.info(f"Found {len(image_paths)} images")
	
	# Ignore all images that already have captions
	image_paths = [path for path in image_paths if not Path(path).with_suffix(".txt").exists()]

	# Load JoyCaption
	tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
	assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"
	llava_model = LlavaForConditionalGeneration.from_pretrained(args.model, torch_dtype="bfloat16", device_map=0)
	assert isinstance(llava_model, LlavaForConditionalGeneration)

	dataset = ImageDataset(prompts, image_paths, tokenizer, llava_model.config.image_token_index, llava_model.config.image_seq_length)
	dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, num_workers=args.num_workers, shuffle=False, drop_last=False, batch_size=args.batch_size)
	end_of_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
	end_of_turn_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
	assert isinstance(end_of_header_id, int) and isinstance(end_of_turn_id, int)

	pbar = tqdm(total=len(image_paths), desc="Captioning images...", dynamic_ncols=True)
	for batch in dataloader:
		vision_dtype = llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
		vision_device = llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
		language_device = llava_model.language_model.get_input_embeddings().weight.device

		# Move to GPU
		pixel_values = batch['pixel_values'].to(vision_device, non_blocking=True)
		input_ids = batch['input_ids'].to(language_device, non_blocking=True)
		attention_mask = batch['attention_mask'].to(language_device, non_blocking=True)

		# Normalize the image
		pixel_values = pixel_values / 255.0
		pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
		pixel_values = pixel_values.to(vision_dtype)

		# Generate the captions
		generate_ids = llava_model.generate(
			input_ids=input_ids,
			pixel_values=pixel_values,
			attention_mask=attention_mask,
			max_new_tokens=args.max_new_tokens,
			do_sample=not args.greedy,
			suppress_tokens=None,
			use_cache=True,
			temperature=args.temperature,
			top_k=args.top_k,
			top_p=args.top_p,
		)

		# Trim off the prompts
		assert isinstance(generate_ids, torch.Tensor)
		generate_ids = generate_ids.tolist()
		generate_ids = [trim_off_prompt(ids, end_of_header_id, end_of_turn_id) for ids in generate_ids]

		# Decode the captions
		captions = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
		captions = [c.strip() for c in captions]

		for path, caption in zip(batch['paths'], captions):
			write_caption(Path(path), args.prepend + caption + args.append)
		pbar.update(len(captions))


def trim_off_prompt(input_ids: list[int], eoh_id: int, eot_id: int) -> list[int]:
	# Trim off the prompt
	while True:
		try:
			i = input_ids.index(eoh_id)
		except ValueError:
			break
		
		input_ids = input_ids[i + 1:]
	
	# Trim off the end
	try:
		i = input_ids.index(eot_id)
	except ValueError:
		return input_ids
	
	return input_ids[:i]


def write_caption(image_path: Path, caption: str):
	caption_path = image_path.with_suffix(".txt")

	try:
		f = os.open(caption_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)  # Write-only, create if not exist, fail if exists
	except FileExistsError:
		logging.warning(f"Caption file '{caption_path}' already exists")
		return
	except Exception as e:
		logging.error(f"Failed to open caption file '{caption_path}': {e}")
		return
	
	try:
		os.write(f, caption.encode("utf-8"))
		os.close(f)
	except Exception as e:
		logging.error(f"Failed to write caption to '{caption_path}': {e}")
		return


class ImageDataset(Dataset):
	def __init__(self, prompts: list[Prompt], paths: list[Path], tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, image_token_id: int, image_seq_length: int):
		self.prompts = prompts
		self.paths = paths
		self.tokenizer = tokenizer
		self.image_token_id = image_token_id
		self.image_seq_length = image_seq_length
		self.pad_token_id = tokenizer.pad_token_id
	
	def __len__(self):
		return len(self.paths)
	
	def __getitem__(self, idx: int) -> dict:
		path = self.paths[idx]

		# Pick a prompt
		prompt_str = random.choices(self.prompts, weights=[p.weight for p in self.prompts])[0].prompt

		# Preprocess image
		# NOTE: I don't use the Processor here and instead do it manually.
		# This is because in my testing a simple resize in Pillow yields higher quality results than the Processor,
		# and the Processor had some buggy behavior on some images.
		# And yes, with the so400m model, the model expects the image to be squished into a square, not padded.
		try:
			image = Image.open(path)
			if image.size != (384, 384):
				image = image.resize((384, 384), Image.LANCZOS)
			image = image.convert("RGB")
			pixel_values = TVF.pil_to_tensor(image)
		except Exception as e:
			logging.error(f"Failed to load image '{path}': {e}")
			pixel_values = None   # Will be filtered out later

		# Build the conversation
		convo = [
			{
				"role": "system",
				"content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.",
			},
			{
				"role": "user",
				"content": prompt_str,
			},
		]

		# Format the conversation
		convo_string = self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
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
		attention_mask = torch.ones_like(input_ids)

		return {
			'path': path,
			'pixel_values': pixel_values,
			'input_ids': input_ids,
			'attention_mask': attention_mask,
		}

	def collate_fn(self, batch: list[dict]) -> dict:
		# Filter out images that failed to load
		batch = [item for item in batch if item['pixel_values'] is not None]

		# Pad input_ids and attention_mask
		# Have to use left padding because HF's generate can't handle right padding it seems
		max_length = max(item['input_ids'].shape[0] for item in batch)
		n_pad = [max_length - item['input_ids'].shape[0] for item in batch]
		input_ids = torch.stack([torch.nn.functional.pad(item['input_ids'], (n, 0), value=self.pad_token_id) for item, n in zip(batch, n_pad)])
		attention_mask = torch.stack([torch.nn.functional.pad(item['attention_mask'], (n, 0), value=0) for item, n in zip(batch, n_pad)])

		# Stack pixel values
		pixel_values = torch.stack([item['pixel_values'] for item in batch])

		# Paths
		paths = [item['path'] for item in batch]

		return {
			'paths': paths,
			'pixel_values': pixel_values,
			'input_ids': input_ids,
			'attention_mask': attention_mask,
		}


def parse_prompts(prompt_str: str | None, prompt_file: str | None) -> list[Prompt]:
	if prompt_str is not None and prompt_file is not None:
		raise ValueError("Cannot specify both --prompt and --prompt-file")

	if prompt_str is not None:
		return [Prompt(prompt=prompt_str, weight=1.0)]
	
	if prompt_file is None:
		raise ValueError("Must specify either --prompt or --prompt-file")
	
	data = json.loads(Path(prompt_file).read_text())

	if not isinstance(data, list):
		raise ValueError("Expected JSON file to contain a list of prompts")
	
	prompts = []

	for item in data:
		if isinstance(item, str):
			prompts.append(Prompt(prompt=item, weight=1.0))
		elif isinstance(item, dict) and "prompt" in item and "weight" in item and isinstance(item["prompt"], str) and isinstance(item["weight"], (int, float)):
			prompts.append(Prompt(prompt=item["prompt"], weight=item["weight"]))
		else:
			raise ValueError(f"Invalid prompt in JSON file. Should be either a string or an object with 'prompt' and 'weight' fields: {item}")

	if len(prompts) == 0:
		raise ValueError("No prompts found in JSON file")

	if sum(p.weight for p in prompts) <= 0.0:
		raise ValueError("Prompt weights must sum to a positive number")

	return prompts


def find_images(glob: str | None, filelist: str | Path | None) -> list[Path]:
	if glob is None and filelist is None:
		raise ValueError("Must specify either --glob or --filelist")

	paths = []

	if glob is not None:
		if PurePath(glob.split('*')[0].split('?')[0]).is_absolute():
			paths.extend(Path(p) for p in glob_module.glob(glob))
		else:
			paths.extend(Path(".").glob(glob))

	if filelist is not None:
		paths.extend((Path(line.strip()) for line in Path(filelist).read_text().strip().splitlines() if line.strip() != ""))

	return paths


if __name__ == "__main__":
	main()
