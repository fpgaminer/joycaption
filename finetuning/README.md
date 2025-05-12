# JoyCaption Fine-tuning

This directory contains the code for fine-tuning the JoyCaption model on your own data.


## Overview

JoyCaption can be finetuned to write captions/descriptions/etc in a specific style or for a specific domain.  For example, if you are building a LORA for Flux/SDXL/etc and want captions for your images, but in a style that JoyCaption doesn't itself generate, you can collect a set of example captions and finetune JoyCaption on those examples.  Then the finetuned JoyCaption can be run on the rest of your images.


## Warning

I have not used a finetuned JoyCaption in production yet.  It is unknown how well this will perform, so consider it experimental.


## Data

The data should be in a JSON format like this:

```
[
  {
    "messages": [
      {
        "role": "user",
        "content": "Respond in JSON, describing the image. Break the image down into different properties like camera angle and framing, subjects, characteristics of each subject, objects, characteristics of each object, background, etc."
      },
      {
        "role": "assistant",
        "content": "```json\n{\n  \"cameraAngle\": \"Straight on\",\n  \"framing\": \"Cowboy shot, capturing the subject from the middle thighs up\",\n  \"subjects\": [\n    {\n      \"type\": \"Man\",\n      \"appearance\": {\n        \"build\": \"Muscular\",\n        \"hair\": \"Short, light brown, styled back\",\n        \"facialHair\": \"Full beard, light brown\",\n        \"chest\": \"Hairy\",\n        \"attire\": {\n          \"top\": \"Shirtless\",\n          \"bottom\": \"Dark blue jeans, unbuttoned and pulled down to reveal black underwear\",\n          \"accessories\": \"Silver chain necklace with pendant\"\n        },\n\t\t\"expression\": \"Smiling, looking at the camera\"\n      }\n    }\n  ],\n  \"objects\": [\n    {\n      \"type\": \"Couch\",\n      \"characteristics\": {\n        \"color\": \"Dark gray\",\n        \"material\": \"Fabric\",\n\t\t\"style\": \"Modern\"\n      }\n    },\n    {\n      \"type\": \"Pillow\",\n      \"characteristics\": {\n        \"shape\": \"Rectangular\",\n        \"color\": \"Cream with gray stripes\",\n        \"material\": \"Fabric with a fuzzy trim\"\n      }\n    },\n    {\n      \"type\": \"Curtains\",\n      \"characteristics\": {\n        \"color\": \"Gray and white\",\n        \"material\": \"Fabric\",\n\t\t\"style\": \"Floor-length\"\n      }\n    },\n\t{\n\t  \"type\": \"Ottoman\",\n\t  \"characteristics\": {\n\t\t\"shape\": \"Rectangular\",\n\t\t\"color\": \"Beige\",\n\t\t\"material\": \"Fabric\",\n\t\t\"style\": \"Tufted\"\n\t  }\n\t}\n  ],\n  \"background\": {\n    \"type\": \"Wall\",\n    \"color\": \"Dark blue\",\n\t\"style\": \"Paneled\"\n  }\n}\n```"
      }
    ],
    "images": [
      "training-images/14780432.jpg"
    ]
  },
  {
    "messages": [
      {
        "role": "user",
        "content": "What's going on in the top left?"
      },
      {
        "role": "assistant",
        "content": "The top left of the image is an inset panel showing an \"X-Ray\" view.  This X-Ray view is typical of pornographic artwork.  In this case it shows the male's penis ejaculating inside of the female character's vaginal canal."
      }
    ],
    "images": [
      "training-images/581402.png"
    ]
  }
]
```

It is an array of examples to train on, with each example consisting of a user message followed by an assistant message, and then the image used for that example.  There must be exactly one image specified for each example.  The images can be any resolution and format; they will be preprocessed by the training script into what JoyCaption needs.  For now, the training script expects exactly one assistant message, and will only train on the assistant's response.  So it does not support multi-turn conversations (yet).


## How much data?

At least 200 examples is likely to give the best results, based on my experience so far.  The exact minimum will vary depending on exactly how much you're adjusting JoyCaption.  Simply changing the style of the captions/descriptions shouldn't require very much data.  Training in new concepts that it must recognize and describe will require more data to ensure consistent and high accuracy.

I recommend that the data is highly diverse.  For example, if you're working on a diffusion LORA, make sure the examples you train JoyCaption on represent all the different types of images you expect to use it on.  If the finetuning data isn't diverse enough, JoyCaption may default back to its original style when it encounters something new.


## What prompt to use?

The training data format lets you put any kind of query from the user, and any kind of response from JoyCaption, so it's quite flexible.  But, if your goal is to keep JoyCaption as a captioner, and just adjust its style, I would recommend using a user prompts similar to the ones JoyCaption expects.  For example "Write a descriptive caption for this image in a formal tone.".  This will maximize the finetuning process, since JoyCaption can build off of its existing knowledge.

If your task is quite different, or if your finetuned models have a tendency to fall back to JoyCaption's default behavior, I recommend using an entirely different user prompt to force the model outside of its comfort zone.  For example, "I want you to write a JSON object with all the details of this image, including camera angle, framing, subjects, etc.".  The user prompt can be any length, so it may also prove helpful to use a very long, detailed prompt to help guide the specifics of how the model should behave.  While JoyCaption itself was not trained on prompts like that, it is based on Llama 3.1, so very long prompts will likely force the model back to its powerful Llama roots, with the benefit of having an image in context.


## Training

Example command:

```
torchrun --standalone --nproc_per_node=1 train.py --wandb-project finetune-2 --device-batch-size 4 --dataset ../instruction-dataset/answers-train.json --max-samples 1800 --images-path ../instruction-dataset --test-every 2000 --test-size 128
```

Check the top of the `train.py` script for a list of all arguments.  The most important parameters are the learning rate, batch size, lora_r, and lora_alpha.  The default settings should work well enough, but they are by no means optimal, just ones I picked and have worked so far.

I recommend adjust test-size based on the size of your dataset.  Usually about 10% of your dataset is a good test size.  At a minimum, the test script runs a test before and after training, so you can assess how well the model has learned your data.  Expect test loss to be around 1.0 when it has learned the task well, though it can be much lower for easy tasks.  If you want to run more tests, you can adjust the `--test-every` parameter to get a measure of test loss throughout training.

I recommend setting `--max-samples` so that about 3 epochs of your data is seen, unless you have a lot of data to train on (10s of thousands or more).  So if you have 200 examples, set `--max-samples` to 600.

You can play around with the various hyperparameters and evaluate the best settings based on which gives you the lowest test loss.  However, be warned, that doing this too much risks "p-hacking" and overfitting.  Generally speaking LORA finetuning like this is quite forgiving, so you shouldn't need to do much if any tuning.


## Advanced Users

This is a very basic script, defaulting to AdamW and basic LORA.  Advanced users may wish to tinker with the code to use a different optimizer, do QLORA, different PEFTs, NEFT tuning, etc.  The training script also uses a fixed system prompt for all examples, since JoyCaption has only seen one system prompt.  But there might be value in adjusting it for specific tasks.


## How to use the finetuned model

Here is an example on how to use the finetuned model.  It's the same as the example from `../README.md`, but uses `PeftModel.from_pretrained()` to load the trained LORA on top of the JoyCaption model.

A word of warning: Make sure you are using the same base model during both training and inference.  Make sure you are using the same prompt(s) during training and inference.  _Any_ deviation, no matter how minor (including whitespace!), can severely impact the performance of the finetuned model.  The training script automatically strips leading and trailing whitespace from prompts and responses to help.

```python
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel


IMAGE_PATH = "image.jpg"
PROMPT = "Please write a question or prompt for this image. The questions or prompts you write are just like what a user might write. The prompt/question should usually be related to the image, but may occasionally not, so as not to bias things. The prompts/questions you write cover the entire range of things users might write, including the entire range of ways users might write, english level, typos, grammar mistakes, etc."
MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"
LORA_PATH = "../joy-caption-finetune/checkpoints/cuu2y0sx/samples_1984/model"


# Load JoyCaption
# bfloat16 is the native dtype of the LLM used in JoyCaption (Llama 3.1)
# device_map=0 loads the model into the first GPU
processor = AutoProcessor.from_pretrained(MODEL_NAME)
llava_model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype="bfloat16", device_map=0)
llava_model = PeftModel.from_pretrained(llava_model, LORA_PATH)
llava_model.eval()

with torch.no_grad():
	# Load image
	image = Image.open(IMAGE_PATH)

	# Build the conversation
	convo = [
		{
			"role": "system",
			"content": "You are a helpful image captioner.",
		},
		{
			"role": "user",
			"content": PROMPT,
		},
	]

	# Format the conversation
	# WARNING: HF's handling of chat's on Llava models is very fragile.  This specific combination of processor.apply_chat_template(), and processor() works
	# but if using other combinations always inspect the final input_ids to ensure they are correct.  Often times you will end up with multiple <bos> tokens
	# if not careful, which can make the model perform poorly.
	convo_string = processor.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
	assert isinstance(convo_string, str)

	# Process the inputs
	inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to('cuda')
	inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

	# Generate the captions
	generate_ids = llava_model.generate(
		**inputs,
		max_new_tokens=300,
		do_sample=True,
		suppress_tokens=None,
		use_cache=True,
		temperature=0.6,
		top_k=None,
		top_p=0.9,
	)[0]

	# Trim off the prompt
	generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

	# Decode the caption
	caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
	caption = caption.strip()
	print(caption)
```


Finetuned models can also be used in vLLM, just like the base model, which provides insanely fast inference and a nice OpenAI compatible API.  However vLLM does not currently support LORAs on VLM type models.  To use your finetune in vLLM you will need to merge the trained LORA weights, for example:

```
model = model.merge_and_unload(progressbar=True)
model.save_pretrained("./questions-cuu2y0sx")
processor.save_pretrained("./questions-cuu2y0sx")
```

and then point vLLM at the merged model:

```bash
vllm serve ./questions-cuu2y0sx --max-model-len 4096 --enable-prefix-caching
```
