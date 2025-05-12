# JoyCaption

JoyCaption is an open, free, and uncensored captioning Visual Language Model (VLM).

[**Try the Demo on HuggingFace**](https://huggingface.co/spaces/fancyfeast/joy-caption-beta-one) **|** [**Download the Current Model on Hugging Face**](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava) **|** [**Latest Release Post**](https://civitai.com/articles/14672)

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

To see JoyCaption in action, check out the [demo on HuggingFace Spaces](https://huggingface.co/spaces/fancyfeast/joy-caption-beta-one).

### Installation

To use JoyCaption locally, you can download the model from [Hugging Face](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava) and integrate it into your existing workflows.

### Example Usage

```python
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


IMAGE_PATH = "image.jpg"
PROMPT = "Write a long descriptive caption for this image in a formal tone."
MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"


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
		max_new_tokens=512,
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

JoyCaption Beta One offers multiple modes of caption generation to suit different needs. `Descriptive Caption` and `Straightforward` are the most useful, with the other modes being interesting but a little less stable. The HuggingFace demo has a nice interface for selecting the output mode and extra options, and it outputs the prompt it used.  Otherwise, here are all the prompts that JoyCaption Beta One understands:

1. **Descriptive Caption**: Writes descriptive captions for the image, either in a formal or casual tone.
   - Examples:
     - "Write a long detailed description for this image."
     - "Write a detailed description for this image in {word_count} words or less."
     - "Write a {length} detailed description for this image."
     - "Write a descriptive caption for this image in a casual tone."
     - "Write a descriptive caption for this image in a casual tone within {word_count} words."
     - "Write a {length} descriptive caption for this image in a casual tone."
   - **Note**: Casual tone can be a bit weird and needs more work, often falling back on a style that is akin to a robot pretending to be "hip" with the youth.

2. **Straightforward Caption**: A more concise, objective style than Descriptive.
   - Examples:
     - "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing."
     - "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing."
     - "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing."

3. **Stable Diffusion Prompt**: Tries to mimic how users typically write Stable Diffusion prompts, with a mixture of natural language and booru-like tags.
   - Examples:
     - "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt."
     - "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less."
     - "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt."
   - **Note**: This mode is more stable than it was in Alpha Two, but can still glitch out ~3% of the time.

4. **MidJourney**: Similar to Training Prompt mode but more like MidJourney prompts.
   - Examples:
     - "Write a MidJourney prompt for this image."
     - "Write a MidJourney prompt for this image within {word_count} words."
     - "Write a {length} MidJourney prompt for this image."
   - **Note**: This mode is still a work in progress and somewhat unstable, occasionally glitching out into a repetition loop (due to limitations of stock Llama 3.1). Use with caution.

5. **Danbooru tag list**: Writes a list of Danbooru tags for the image.
   - Examples:
     - "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text."
     - "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less."
     - "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length."
   - **Note**: This mode has lower accuracy and overall performance than the other modes.

6. **e621 tag list**: Writes a list of e621 tags for the image.
   - Examples:
	 - "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags."
	 - "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words."
	 - "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags."
   - **Note**: This mode has lower accuracy and overall performance than the other modes.

7. **Rule34 tag list**: Writes a list of Rule34 tags for the image.
   - Examples:
	 - "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags."
     - "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words."
     - "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags."
   - **Note**: This mode has lower accuracy and overall performance than the other modes.

8. **Booru-Like Tag List**: Similar to Booru Tag List mode, but will write outside the strict list of tags that boorus use.
   - Examples:
     - "Write a list of Booru-like tags for this image."
     - "Write a list of Booru-like tags for this image within {word_count} words."
     - "Write a {length} list of Booru-like tags for this image."
   - **Note**: This mode is still a work in progress and somewhat unstable, occasionally glitching out into a repetition loop (due to limitations of stock Llama 3.1). Use with caution.

9. **Art Critic Analysis**: Writes an analysis of the image like an art critic.
   - Examples:
     - "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc."
     - "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words."
     - "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}."

10. **Product Listing**: Writes a product listing-style caption for the image.
   - Examples:
     - "Write a caption for this image as though it were a product listing."
     - "Write a caption for this image as though it were a product listing. Keep it under {word_count} words."
     - "Write a {length} caption for this image as though it were a product listing."

11. **Social Media Post**: Writes a caption for the image suitable for a social media post.
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
- If it is a work of art, do not include the artist's name or the title of the work.
- Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.
- Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.
- Do NOT use polite euphemisms—lean into blunt, casual phrasing.
- Include information about the ages of any people/characters when applicable.
- Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.
- Do not mention the mood/feeling/etc of the image.
- Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.).
- If there is a watermark, you must mention it.
- Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…”, "You are looking at...", etc.


### Limitations

**WARNING:** Beta One is not a general instruction follower.  Feel free to experiment outside of these prompts, but don't expect perfect adherence.


## vLLM

vLLM provides the highest performance inference for JoyCaption, and an OpenAI compatible API so JoyCaption can be used like any other VLMs. Example usage:

```bash
vllm serve fancyfeast/llama-joycaption-beta-one-hf-llava --max-model-len 4096 --enable-prefix-caching
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

JoyCaption is currently at Beta One. This means that things are nearing completion for version 1.0.

Please note that JoyCaption, like all VLMs, is not perfect.  Expect issues when it comes to multiple subjects, left/right confusion, OCR inaccuracy, etc.  Instruction following is better than Alpha Two, but will occasionally fail and is not as robust as a fully fledged SOTA VLM.  And though I've drastically reduced the incidence of glitches, they do still occur 1.5 to 3% of the time.  As an independent developer, I'm limited in how far I can push things.  For comparison, commercial models like GPT4o have a glitch rate of 0.01%.

If you use Beta One as a more general purpose VLM, asking it questions and such, on NSFW queries you may find that it _occasionally_ responds with a refusal.  This is not intentional, and Beta One itself was not censored.  However certain queries can trigger llama's old safety behavior.  Simply re-try the question, phrase it differently, or tweak the system prompt to get around this.


## Feedback and Contributions

Feedback is always welcome and crucial to helping me improve JoyCaption for everyone to use! If you have suggestions for improvement, notice weaknesses, or want to contribute to the project, please reach out.


## Release history

* Pre-Alpha: https://www.reddit.com/r/StableDiffusion/comments/1egwgfk/joycaption_free_open_uncensored_vlm_early/
* Alpha One: https://www.reddit.com/r/StableDiffusion/comments/1fm9pxa/joycaption_free_open_uncensored_vlm_alpha_one/
* Alpha Two: https://civitai.com/articles/7697
* Beta One: https://civitai.com/articles/14672
