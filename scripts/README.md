Documentation for the scripts in the `scripts` directory, starting with `batch-caption.py`, which is used to run JoyCaption in bulk. Other scripts might be added in the future.

# batch-caption.py

## Basic Command

To run the script, use the following command:

```sh
./batch-caption.py --glob "path/to/images/*.jpg" --prompt "Write a descriptive caption for this image in a formal tone."
```

This command will caption all the `.jpg` images in the relative or absolute directory using the provided prompt, writing `.txt` files alongside each image.


## Command-Line Arguments

**Note**: You must specify either `--glob` or `--filelist` to provide images, and either `--prompt` or `--prompt-file` to provide a prompt for caption generation.

| Argument           | Description                                                | Default                                                                                                                 |
| ------------------ | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `--glob`           | Glob pattern to find images                                | N/A                                                                                                                     |
| `--filelist`       | File containing a list of images                           | N/A                                                                                                                     |
| `--prompt`         | Prompt to use for caption generation                       | N/A                                                                                                                     |
| `--prompt-file`    | JSON file containing prompts                               | N/A                                                                                                                     |
| `--batch-size`     | Batch size for image processing                            | 1                                                                                                                       |
| `--greedy`         | Use greedy decoding instead of sampling                    | False                                                                                                                   |
| `--temperature`    | Sampling temperature (used when not using greedy decoding) | 0.6                                                                                                                     |
| `--top-p`          | Top-p sampling value (nucleus sampling)                    | 0.9                                                                                                                     |
| `--top-k`          | Top-k sampling value                                       | None                                                                                                                    |
| `--max-new-tokens` | Maximum length of the generated caption (in tokens)        | 256                                                                                                                     |
| `--num-workers`    | Number of workers loading images in parallel               | 4                                                                                                                       |
| `--model`          | Pre-trained model to use                                   | [fancyfeast/llama-joycaption-alpha-two-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava) |



### Examples

1. **Caption images with a specific prompt**

   ```sh
   ./batch-caption.py --glob "images/*.png" --prompt "Write a descriptive caption for this image in a formal tone."
   ```

2. **Use a JSON file for prompts**

   ```sh
   python batch-caption.py --filelist "image_paths.txt" --prompt-file "prompts.json"
   ```

3. **Use Greedy Decoding**

   ```sh
   python batch-caption.py --glob "images/*.jpg" --prompt "Write a descriptive caption for this image in a formal tone." --greedy
   ```

## Prompt Handling

- For a list of prompts that the model understands, please refer to the project's root README.

- You can specify a prompt directly using the `--prompt` argument or use a JSON file containing a list of prompts with weights using `--prompt-file`.

- If multiple prompts are specified in the prompt file, the prompt used for each image will be randomly selected.

- **Prompt File Format**: The JSON file should contain either strings or objects with `prompt` and `weight` fields.

  - **Weighting**: The `weight` field indicates the probability of selecting a particular prompt during caption generation. Higher weights make a prompt more likely to be chosen. For example, if one prompt has a weight of 2.0 and another has a weight of 1.0, the first prompt will be twice as likely to be used.

Example `prompts.json`:

```json
[
  { "prompt": "Describe the scene in detail.", "weight": 2.0 },
  { "prompt": "Summarize the main elements of the image.", "weight": 1.0 }
]
```

## Output

- Captions are saved as `.txt` files in the same directory as the corresponding image.
- If a `.txt` caption file already exists for an image, the script will skip that image.
