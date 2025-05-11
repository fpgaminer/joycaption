# JoyCaption

JoyCaption is an open, free, and uncensored captioning Visual Language Model (VLM).

[**Try the Demo on HuggingFace**](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two) **|** [**Download the Current Model on Hugging Face**](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava) **|** [**Latest Release Post**](https://civitai.com/articles/7697)

![This image is a digital meme featuring a photograph of a golden retriever walking on a wet asphalt road. The dog, with a light golden coat, wears a red collar and leash, and is captured mid-stride, with its tongue out. The background is blurred, emphasizing the dog. Above the dog, in bold black text on a white background, it reads, "Self-supervised Learning." The overall tone is humorous, combining the concept of self-supervised learning with the playful image of the dog.](dog.jpg)


## What is JoyCaption?

JoyCaption is an image captioning Visual Language Model (VLM) being built from the ground up as a free, open, and uncensored model for the community to use in training Diffusion models.

Key Features:
- **Free and Open**: Released for free, open weights, no restrictions, and just like [bigASP](https://www.reddit.com/r/StableDiffusion/comments/1dbasvx/the_gory_details_of_finetuning_sdxl_for_30m/), will come with training scripts and lots of juicy details on how it gets built.
- **Uncensored**: Equal coverage of SFW and NSFW concepts. No "cylindrical shaped object with a white substance coming out on it" here.
- **Diversity**: All are welcome here. Do you like digital art? Photoreal? Anime? Furry? JoyCaption is for everyone. Pains are being taken to ensure broad coverage of image styles, content, ethnicity, gender, orientation, etc.
- **Minimal Filtering**: JoyCaption is trained on large swathes of images so that it can understand almost all aspects of our world. almost. Illegal content will never be tolerated in JoyCaption's training.


## Motivation

Automated descriptive captions enable the training and finetuning of diffusion models on a wider range of images, since trainers are no longer required to either find images with already associated text or write the descriptions themselves. They also improve the quality of generations produced by Text-to-Image models trained on them (ref: DALL-E 3 paper). But to-date, the community has been stuck with ChatGPT, which is expensive and heavily censored; or alternative models, like CogVLM, which are weaker than ChatGPT and have abysmal performance outside of the SFW domain.

I'm building JoyCaption to help fill this gap by performing near or on-par with GPT4o in captioning images, while being free, unrestricted, and open.


## Using JoyCaption

### Demo

To see JoyCaption in action, check out the [demo on HuggingFace Spaces](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two).

### Installation

To use JoyCaption locally, you can download the model from [Hugging Face](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava) and integrate it into your existing workflows.

### Example Usage

```python
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


IMAGE_PATH = "image.jpg"
PROMPT = "Write a long descriptive caption for this image in a formal tone."
MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"


# Load JoyCaption
# bfloat16 is the native dtype of the LLM used in JoyCaption (Llama 3.1)
# device_map=0 loads the model into the first GPU
processor = AutoProcessor.from_pretrained(MODEL_NAME)
llava_model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype="bfloat16", device_map=0)
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


## How to Prompt JoyCaption

JoyCaption Alpha Two offers multiple modes of caption generation to suit different needs. `Descriptive Caption` prompting is the most useful, with the other modes being _experimental_. The HuggingFace demo has a nice interface for selecting the output mode and extra options, and it outputs the prompt it used.  Otherwise, here are all the prompts that JoyCaption Alpha Two understands:

1. **Descriptive Caption**: Writes descriptive captions for the image, either in a formal or informal tone.
   - Examples:
     - "Write a descriptive caption for this image in a formal tone."
     - "Write a descriptive caption for this image in a formal tone within {word_count} words."
     - "Write a {length} descriptive caption for this image in a formal tone."
     - "Write a descriptive caption for this image in a casual tone."
     - "Write a descriptive caption for this image in a casual tone within {word_count} words."
     - "Write a {length} descriptive caption for this image in a casual tone."
   - **Note**: Informal tone is ... weird and experimental at the moment. It helps expand the vocabulary of the captions and writing styles, which could be helpful, but yeah ... expect everything to be "vibes" and "rocking". 😒

2. **Training Prompt**: Writes more like the average Stable Diffusion prompt, with a mixture of natural language and booru-like tags, mimicing what users might prompt SD to get the image.
   - Examples:
     - "Write a stable diffusion prompt for this image."
     - "Write a stable diffusion prompt for this image within {word_count} words."
     - "Write a {length} stable diffusion prompt for this image."
   - **Note**: This mode is still a work in progress and somewhat unstable, occasionally glitching out into a repetition loop (due to limitations of stock Llama 3.1). Use with caution.

3. **MidJourney**: Similar to Training Prompt mode but more like MidJourney prompts.
   - Examples:
	 - "Write a MidJourney prompt for this image."
     - "Write a MidJourney prompt for this image within {word_count} words."
     - "Write a {length} MidJourney prompt for this image."
   - **Note**: This mode is still a work in progress and somewhat unstable, occasionally glitching out into a repetition loop (due to limitations of stock Llama 3.1). Use with caution.

4. **Booru Tag List**: Writes a list of Booru-style tags for the image.
   - Examples:
     - "Write a list of Booru tags for this image."
     - "Write a list of Booru tags for this image within {word_count} words."
     - "Write a {length} list of Booru tags for this image."
   - **Note**: This mode is still a work in progress and somewhat unstable, occasionally glitching out into a repetition loop (due to limitations of stock Llama 3.1). Use with caution.

5. **Booru-Like Tag List**: Similar to Booru Tag List mode, but will write outside the strict list of tags that boorus use.
   - Examples:
     - "Write a list of Booru-like tags for this image."
     - "Write a list of Booru-like tags for this image within {word_count} words."
     - "Write a {length} list of Booru-like tags for this image."
   - **Note**: This mode is still a work in progress and somewhat unstable, occasionally glitching out into a repetition loop (due to limitations of stock Llama 3.1). Use with caution.

6. **Art Critic Analysis**: Writes an analysis of the image like an art critic.
   - Examples:
     - "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc."
     - "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words."
     - "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}."

7. **Product Listing**: Writes a product listing-style caption for the image.
   - Examples:
     - "Write a caption for this image as though it were a product listing."
     - "Write a caption for this image as though it were a product listing. Keep it under {word_count} words."
     - "Write a {length} caption for this image as though it were a product listing."

8. **Social Media Post**: Writes a caption for the image suitable for a social media post.
   - Examples:
     - "Write a caption for this image as if it were being used for a social media post."
     - "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words."
     - "Write a {length} caption for this image as if it were being used for a social media post."


### Extra Options

The following extra instructions can be appended to the prompt to guide the caption generation:

- If there is a person/character in the image you must refer to them as {name}.
- Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).
- Include information about lighting.
- Include information about camera angle.
- Include information about whether there is a watermark or not.
- Include information about whether there are JPEG artifacts or not.
- If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.
- Do NOT include anything sexual; keep it PG.
- Do NOT mention the image's resolution.
- You MUST include information about the subjective aesthetic quality of the image from low to very high.
- Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.
- Do NOT mention any text that is in the image.
- Specify the depth of field and whether the background is in focus or blurred.
- If applicable, mention the likely use of artificial or natural lighting sources.
- Do NOT use any ambiguous language.
- Include whether the image is sfw, suggestive, or nsfw.
- ONLY describe the most important elements of the image.


### Limitations

**WARNING:** Alpha Two was heavily trained on the above Prompts and Extra Options.  It is not a general instruction follower.  Feel free to experiment outside of these prompts, but don't expect great results (yet).


## vLLM

vLLM provides the highest performance inference for JoyCaption, and an OpenAI compatible API so JoyCaption can be used like any other VLMs. Example usage:

```bash
vllm serve fancyfeast/llama-joycaption-alpha-two-hf-llava --max-model-len 4096 --enable-prefix-caching
```

VLMs are a bit finicky on vLLM, and vLLM is memory hungry, so you may have to adjust settings for your particular environment, such as forcing eager mode, adjusting max-model-len, adjusting gpu_memory_utilization, etc.

On Windows the easiest way to use vLLM is to setup docker and run something like:

```bash
docker run --gpus all --ipc=host -p 8000:8000 -v "%USERPROFILE%\.cache\huggingface:/root/.cache/huggingface" vllm/vllm-openai:latest --model fancyfeast/llama-joycaption-beta-one-hf-llava --max-model-len 4096 --enable-prefix-caching
```

`-v "%USERPROFILE%\.cache\huggingface:/root/.cache/huggingface"` persists the model cache outside of docker, so it doesn't have to re-download the model every time you run the container.


## Finetuning

Finetuning scripts and documentation can be found in the `finetuning` directory.  The `finetuning/README.md` file contains detailed instructions on how to prepare your data and train JoyCaption on it.


## Current Status

JoyCaption is currently at Alpha Two. This means that it is still under development, and improvements are continuously being made based on feedback from users.

Please note that JoyCaption is not yet ready for production use. It's an experimental release, and you may encounter mistakes, especially when it comes to interactions between characters in an image, OCR, and confusing left/right when describing objects and actions in releation to people.


## Feedback and Contributions

Feedback is always welcome and crucial to helping me improve JoyCaption for everyone to use! If you have suggestions for improvement, notice weaknesses, or want to contribute to the project, please reach out.


## Release history

* Pre-Alpha: https://www.reddit.com/r/StableDiffusion/comments/1egwgfk/joycaption_free_open_uncensored_vlm_early/
* Alpha One: https://www.reddit.com/r/StableDiffusion/comments/1fm9pxa/joycaption_free_open_uncensored_vlm_alpha_one/
* Alpha Two: https://civitai.com/articles/7697
