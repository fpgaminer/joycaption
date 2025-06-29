import gradio as gr
from transformers import LlavaForConditionalGeneration, TextIteratorStreamer, AutoProcessor
import torch
from PIL import Image
from threading import Thread
from typing import Generator
# from liger_kernel.transformers import apply_liger_kernel_to_llama
from pathlib import Path
from gradio.utils import NamedString
import zipfile
import tempfile
import traceback
from functools import partial
from tqdm import tqdm
import gc
from importlib import metadata
import platform
from textwrap import indent
import sys




LOGO_SRC = """data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+Cjxzdmcgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgdmlld0JveD0iMCAwIDUzOCA1MzUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSIgeG1sbnM6c2VyaWY9Imh0dHA6Ly93d3cuc2VyaWYuY29tLyIgc3R5bGU9ImZpbGwtcnVsZTpldmVub2RkO2NsaXAtcnVsZTpldmVub2RkO3N0cm9rZS1saW5lam9pbjpyb3VuZDtzdHJva2UtbWl0ZXJsaW1pdDoyOyI+CiAgICA8ZyB0cmFuc2Zvcm09Im1hdHJpeCgxLDAsMCwxLC0xNDcuODcxLDAuMDAxOTA4NjMpIj4KICAgICAgICA8cGF0aCBkPSJNMTk1LjY3LDIyMS42N0MxOTYuNzMsMjA1LjM3IDIwMC4yOCwxODkuNzYgMjA3LjkxLDE3NS4zN0MyMjcuOTgsMTM3LjUxIDI1OS4zMywxMTQuODggMzAyLjAxLDExMS42M0MzMzQuMTUsMTA5LjE4IDM2Ni41OSwxMTAuNiAzOTguODksMTEwLjNDNDAwLjUzLDExMC4yOCA0MDIuMTYsMTEwLjMgNDA0LjQsMTEwLjNDNDA0LjQsMTAxLjk5IDQwNC41Niw5NC4wNSA0MDQuMjMsODYuMTJDNDA0LjE4LDg0Ljg0IDQwMi4xNSw4My4xMyA0MDAuNjYsODIuNDlDMzgzLjIzLDc1LjAyIDM3My4wNSw1OS43OSAzNzMuOTYsNDAuOTZDMzc1LjA5LDE3LjU0IDM5MS40NywyLjY2IDQxMC42NSwwLjM3QzQzNy44OSwtMi44OSA0NTUuNTYsMTUuODQgNDU5LjI2LDM0LjY5QzQ2Mi45Niw1My41NyA0NTIuMTgsNzYuOTMgNDMyLjgxLDgyLjY2QzQzMS42NCw4My4wMSA0MzAuMzMsODUuMjMgNDMwLjI4LDg2LjYyQzQzMC4wMyw5NC4yNiA0MzAuMTYsMTAxLjkyIDQzMC4xNiwxMTAuM0w0MzUuNjMsMTEwLjNDNDYzLjc5LDExMC4zIDQ5MS45NiwxMTAuMjggNTIwLjEyLDExMC4zQzU3NC44NCwxMTAuMzYgNjIzLjA0LDE0OC4zNSA2MzUuNjcsMjAxLjU1QzYzNy4yMywyMDguMTMgNjM3LjgzLDIxNC45MyA2MzguODksMjIxLjY3QzY2MC40MywyMjQuOTQgNjc1LjE5LDIzNi42MiA2ODIuMzYsMjU3LjRDNjgzLjU5LDI2MC45NyA2ODQuNjUsMjY0LjgyIDY4NC42NywyNjguNTRDNjg0Ljc3LDI4My4zNCA2ODUuNzYsMjk4LjMxIDY4My45NCwzMTIuOTFDNjgwLjg5LDMzNy4yOSA2NjIuODYsMzUzLjM2IDYzOC40NywzNTUuODJDNjM1LjE0LDM4NS4wOCA2MjEuOTEsNDA5LjQxIDYwMC40NSw0MjkuMjFDNTgxLjYsNDQ2LjYxIDU1OS4xNCw0NTcuNSA1MzMuNTcsNDU5LjE4QzUwOC4xOCw0NjAuODQgNDgyLjY0LDQ2MC4yIDQ1Ny4xNiw0NjAuMzhDNDM1LjE2LDQ2MC41MyA0MTMuMTcsNDYwLjM0IDM5MS4xNyw0NjAuNTNDMzg4Ljc2LDQ2MC41NSAzODUuOTUsNDYxLjU2IDM4NC4wMyw0NjMuMDRDMzcxLjU0LDQ3Mi42MiAzNTkuMTMsNDgyLjMxIDM0Ni45Miw0OTIuMjVDMzM4Ljk0LDQ5OC43NSAzMzEuMzksNTA1Ljc3IDMyMy41Niw1MTIuNDZDMzE3LjQ1LDUxNy42OCAzMTAuOTMsNTIyLjQ0IDMwNS4xMSw1MjcuOTVDMzAxLjE5LDUzMS42NiAyOTYuNTIsNTMzLjE3IDI5MS42OSw1MzQuMzZDMjg1LjY1LDUzNS44NSAyNzkuMjIsNTI5LjEzIDI3OS4wMSw1MjEuMTlDMjc4LjgsNTEyLjg2IDI3OC45NSw1MDQuNTMgMjc4Ljk0LDQ5Ni4xOUwyNzguOTQsNDU2LjY5QzIzMi44Miw0MzguMTYgMjAzLjU2LDQwNi4yMyAxOTUuMDcsMzU2LjA4QzE5My4yNiwzNTUuNzUgMTkwLjg0LDM1NS40MSAxODguNDgsMzU0Ljg2QzE2Ny40NiwzNDkuOTEgMTU1LjA0LDMzNi4wMiAxNTAuNzIsMzE1LjYyQzE0Ni45OCwyOTcuOTkgMTQ2LjksMjc5LjY3IDE1MC42MSwyNjIuMDlDMTU1LjU1LDIzOC42OCAxNzEuNDIsMjI1LjU5IDE5NS42NiwyMjEuNjdMMTk1LjY3LDIyMS42N1pNMzA4LjA3LDQ4Ny44MkMzMTUuOTQsNDgxLjEzIDMyMi44NSw0NzUuMTMgMzI5LjksNDY5LjNDMzQ0LjM5LDQ1Ny4zMSAzNTguOSw0NDUuMzYgMzczLjU0LDQzMy41NkMzNzUuMTcsNDMyLjI1IDM3Ny42OCw0MzEuNCAzNzkuNzksNDMxLjM5QzQxNC43OCw0MzEuMjYgNDQ5Ljc4LDQzMS4zOCA0ODQuNzcsNDMxLjI0QzUwMC4zOSw0MzEuMTggNTE2LjEzLDQzMS43NiA1MzEuNjIsNDMwLjE2QzU3Ni45Miw0MjUuNDkgNjA5LjI0LDM4Ny43NyA2MDguOTUsMzQ0Ljg0QzYwOC42OCwzMDUuNTIgNjA4LjkzLDI2Ni4xOSA2MDguODcsMjI2Ljg2QzYwOC44NywyMjMuMjIgNjA4LjU4LDIxOS41NSA2MDcuOTksMjE1Ljk2QzYwMy4xMSwxODYuMjkgNTg4LjYxLDE2My4zMyA1NjEuMzIsMTQ5LjMyQzU0OS4wNCwxNDMuMDIgNTM2LjE1LDEzOS4yOSA1MjIuMjIsMTM5LjI5QzQ1My45LDEzOS4zMiAzODUuNTgsMTM5LjIgMzE3LjI2LDEzOS4zNUMzMDkuMiwxMzkuMzcgMzAwLjk2LDEzOS44OSAyOTMuMTEsMTQxLjZDMjU0LjE5LDE1MC4wNyAyMjUuMzMsMTg1LjY5IDIyNS4wMywyMjUuNDJDMjI0LjgsMjU2LjA4IDIyNC44NiwyODYuNzQgMjI0Ljk5LDMxNy40QzIyNS4wNSwzMzAuNTMgMjI0Ljc0LDM0My43NiAyMjYuMTgsMzU2Ljc3QzIyOC43NCwzODAuMDUgMjQwLjYsMzk4LjYyIDI1OC43OSw0MTIuOTNDMjczLjA0LDQyNC4xNCAyODkuNjMsNDMwLjAyIDMwNy42MSw0MzEuNTVDMzA3LjgyLDQzMi4wMyAzMDguMDYsNDMyLjMzIDMwOC4wNiw0MzIuNjNDMzA4LjA4LDQ1MC42IDMwOC4wOCw0NjguNTcgMzA4LjA4LDQ4Ny44MUwzMDguMDcsNDg3LjgyWk00MzUuNzksNDMuMzNDNDM1Ljk1LDMzLjQyIDQyNy42MSwyNC42NSA0MTcuOCwyNC40QzQwNi43NiwyNC4xMiAzOTguMjUsMzIuMDUgMzk4LjEzLDQyLjc0QzM5OC4wMSw1My4wNCA0MDYuNiw2Mi4xMiA0MTYuNDIsNjIuMDhDNDI3LjExLDYyLjA0IDQzNS42MSw1My44MSA0MzUuNzgsNDMuMzNMNDM1Ljc5LDQzLjMzWiIgc3R5bGU9ImZpbGw6cmdiKDczLDQ3LDExOCk7ZmlsbC1ydWxlOm5vbnplcm87Ii8+CiAgICAgICAgPHBhdGggZD0iTTQxOS4zLDM5MS42M0MzNzQuNDYsMzkwLjQgMzQxLjUxLDM3Mi42MyAzMTguMDEsMzM3LjcxQzMxNS42NywzMzQuMjMgMzEzLjc3LDMzMC4wNCAzMTMuMSwzMjUuOTVDMzExLjg0LDMxOC4yOCAzMTYuNTMsMzExLjcgMzIzLjcyLDMwOS40NkMzMzAuNjYsMzA3LjI5IDMzOC4zMiwzMTAuMSAzNDEuOTgsMzE3LjAzQzM0OS4xNSwzMzAuNjMgMzU5LjE2LDM0MS4zNSAzNzIuMywzNDkuMzFDNDAxLjMyLDM2Ni44OSA0NDQuNTYsMzYzLjcgNDcwLjYxLDM0Mi4zNUM0NzkuMSwzMzUuMzkgNDg2LjA4LDMyNy40MSA0OTEuNTUsMzE3Ljk3QzQ5NS4wNSwzMTEuOTMgNTAwLjIsMzA4LjE4IDUwNy40NywzMDguOTVDNTEzLjczLDMwOS42MSA1MTguODYsMzEyLjg4IDUyMC4xMiwzMTkuMjFDNTIwLjksMzIzLjEzIDUyMC43MywzMjguMjIgNTE4LjgzLDMzMS41NUM1MDAuNjMsMzYzLjMyIDQ3My41NSwzODIuOTUgNDM3LjI5LDM4OS4zN0M0MzAuNDQsMzkwLjU4IDQyMy40OCwzOTEuMTIgNDE5LjI5LDM5MS42M0w0MTkuMywzOTEuNjNaIiBzdHlsZT0iZmlsbDpyZ2IoMjUwLDEzOSwxKTtmaWxsLXJ1bGU6bm9uemVybzsiLz4KICAgICAgICA8cGF0aCBkPSJNNDYyLjcxLDI0MC4xOUM0NjIuOCwyMTYuOTEgNDgwLjI0LDE5OS43OSA1MDQuMDEsMTk5LjY3QzUyNi41NywxOTkuNTUgNTQ0Ljg5LDIxOC4wNyA1NDQuNTEsMjQxLjM0QzU0NC4xOCwyNjEuODUgNTMwLjA5LDI4MS45NiA1MDEuOTEsMjgxLjIzQzQ4MC42OCwyODAuNjggNDYyLjE1LDI2My44IDQ2Mi43MSwyNDAuMkw0NjIuNzEsMjQwLjE5WiIgc3R5bGU9ImZpbGw6cmdiKDI1MCwxMzksMSk7ZmlsbC1ydWxlOm5vbnplcm87Ii8+CiAgICAgICAgPHBhdGggZD0iTTM3MC45OSwyNDAuMDhDMzcxLDI2Mi43OSAzNTIuNTMsMjgxLjM1IDMyOS44OSwyODEuMzdDMzA3LjA1LDI4MS40IDI4OC45NiwyNjMuNDIgMjg4Ljk2LDI0MC42OEMyODguOTYsMjE4LjE0IDMwNi43MywyMDAgMzI5LjE2LDE5OS42MkMzNTIuMDIsMTk5LjI0IDM3MC45OCwyMTcuNTcgMzcwLjk5LDI0MC4wOFoiIHN0eWxlPSJmaWxsOnJnYigyNTAsMTM5LDEpO2ZpbGwtcnVsZTpub256ZXJvOyIvPgogICAgPC9nPgo8L3N2Zz4K"""

MODEL_PATH = "fancyfeast/llama-joycaption-beta-one-hf-llava"
TITLE = f"""<style>
  .joy-header   {{display:flex; align-items:center; justify-content:center;
                 gap:16px; margin:4px 0 12px;}}
  .joy-header h1{{margin:0; font-size:1.9rem; line-height:1.2;}}
  .joy-header p {{margin:2px 0 0; font-size:0.9rem; color:#666;}}
  .joy-header img{{height:56px;}}

  .error-box {{
    border: 1px solid #ff5555;
    background-color: #fff0f0;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 8px 0;
    color: #d00;
    font-size: 0.9em;
  }}
  
  .info-box {{
    border: 1px solid #5555ff;
    background-color: #f0f0ff;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 8px 0;
    color: #00d;
    font-size: 0.9em;
  }}
  
  .success-box {{
    border: 1px solid #55aa55;
    background-color: #f0fff0;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 8px 0;
    color: #080;
    font-size: 0.9em;
  }}
  
  .panel-header {{
    background-color: #f5f5f5;
    padding: 10px;
    margin-bottom: 15px;
    border-radius: 4px;
    border-left: 4px solid #4b5563;
  }}
  
  .compact-row {{
    margin-bottom: 0 !important;
  }}

  #global_error {{
    background-color: #fff0f0;
    border-left: 4px solid #ff5555;
    padding: 12px;
    margin: 0 0 15px 0;
    font-weight: 500;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  }}
</style>

<div class="joy-header">
  <img src="{LOGO_SRC}" alt="JoyCaption logo">
  <div>
    <h1>JoyCaption <span style="font-weight:400">Beta&nbsp;One</span></h1>
    <p>Image-captioning model &nbsp;|&nbsp; build mb3500zp</p>
  </div>
</div>
<hr>"""
DESCRIPTION = """
<div>
<h2>Quick-start (Single-Image tab)</h2>
<ol>
  <li><strong>Upload</strong> an image in the left-hand panel.</li>
  <li>Select a <strong>Caption&nbsp;Type</strong> and set a <strong>Caption&nbsp;Length</strong>
      (or leave “any”).</li>
  <li>(Optional) open <em>Extra&nbsp;Options</em> and tick anything
      you want the model to mention / omit.</li>
  <li>(Optional) open <em>Generation settings</em> to adjust
      <code>temperature</code>, <code>top-p</code>, or <code>max tokens</code>.</li>
  <li>Press <kbd>Caption</kbd>.  
      The exact prompt goes into the <em>Prompt</em> box (editable);  
      the caption streams into the <em>Generated Caption</em> box.</li>
</ol>

<h2>Quick-start (Batch tab)</h2>
<ol>
  <li><strong>Upload multiple images</strong> (PNG/JPEG/WEBP, any mix).
      Filenames are used to name the output <code>.txt</code> files.</li>
  <li>Set <strong>DataLoader Workers</strong> (CPU processes) and
      <strong>Batch Size</strong> to balance speed vs. GPU VRAM.</li>
  <li>Press <kbd>Start Batch Process & Create ZIP</kbd>.
      When finished, a ZIP containing all captions appears for download.</li>
</ol>

<!-- ───────────────────── Caption-type reference ──────────────────── -->
<h2>Caption Types</h2>
<table>
  <tr><th>Mode</th><th>What it does</th></tr>
  <tr><td><strong>Descriptive</strong></td>
      <td>Formal, detailed prose description.</td></tr>
  <tr><td><strong>Descriptive&nbsp;(Casual)</strong></td>
      <td>Similar to Descriptive but with a friendlier, conversational tone.</td></tr>
  <tr><td><strong>Straightforward</strong></td>
      <td>Objective, no fluff, and more succinct than Descriptive.</td></tr>
  <tr><td><strong>Stable Diffusion Prompt</strong></td>
      <td>Reverse-engineers a prompt that could have produced the image in a SD/T2I model.<br><em>⚠︎ Experimental – can glitch ≈ 3% of the time.</em></td></tr>
  <tr><td><strong>MidJourney</strong></td>
      <td>Same idea as above but tuned to MidJourney’s prompt style.<br><em>⚠︎ Experimental – can glitch ≈ 3% of the time.</em></td></tr>
  <tr><td><strong>Danbooru tag list</strong></td>
      <td>Comma-separated tags strictly following Danbooru conventions
          (artist:, copyright:, etc.). Lower-case underscores only.<br><em>⚠︎ Experimental – can glitch ≈ 3% of the time.</em></td></tr>
  <tr><td><strong>e621 tag list</strong></td>
      <td>Alphabetical, namespaced tags in e621 style – includes species/meta
          tags when relevant.<br><em>⚠︎ Experimental – can glitch ≈ 3% of the time.</em></td></tr>
  <tr><td><strong>Rule34 tag list</strong></td>
      <td>Rule34 style alphabetical tag dump; artist/copyright/character
          prefixes first.<br><em>⚠︎ Experimental – can glitch ≈ 3% of the time.</em></td></tr>
  <tr><td><strong>Booru-like tag list</strong></td>
      <td>Looser tag list when you want labels but not a specific Booru format.<br><em>⚠︎ Experimental – can glitch ≈ 3% of the time.</em></td></tr>
  <tr><td><strong>Art Critic</strong></td>
      <td>Paragraph of art-historical commentary: composition, symbolism, style,
          lighting, movement, etc.</td></tr>
  <tr><td><strong>Product Listing</strong></td>
      <td>Short marketing copy as if selling the depicted object.</td></tr>
  <tr><td><strong>Social Media Post</strong></td>
      <td>Catchy caption aimed at platforms like Instagram or BlueSky.</td></tr>
  <tr><td><strong>Custom Photo</strong></td>
      <td>Photographic description with camera details.</td></tr>
</table>

<p style="margin-top:0.6em">
  <strong>Note&nbsp;on Booru modes:</strong> They’re tuned for
  anime-style / illustration imagery; accuracy drops on real-world photographs
  or highly abstract artwork.
</p>

<!-- ───────────────────── Extras + generation notes ───────────────── -->
<h3>Extra Options</h3>
<p>These check-boxes fine-tune what the model should or should not mention:
lighting, camera angle, aesthetic rating, profanity, etc.  
Toggle them before hitting <kbd>Caption</kbd>; the prompt box will update
instantly.</p>

<h3>Generation settings</h3>
<ul>
  <li><strong>Temperature</strong> – randomness.  
      0&nbsp;=&nbsp;deterministic; higher =&nbsp;more variety.</li>
  <li><strong>Top-p</strong> – nucleus sampling cutoff. Lower =&nbsp;safer,
      higher =&nbsp;freer.</li>
  <li><strong>Max&nbsp;New Tokens</strong> – hard stop for the model’s output length.</li>
</ul>

<h2>Model Quantization</h2>
<p>Select <kbd>bf16</kbd>, <kbd>8-bit</kbd>, or <kbd>nf4 (4-bit)</kbd>.
Lower precision uses ~⅓–½ the VRAM but may degrade quality slightly.</p>

</div>
"""

CAPTION_TYPE_MAP = {
	"Descriptive": [
		"Write a detailed description for this image.",
		"Write a detailed description for this image in {word_count} words or less.",
		"Write a {length} detailed description for this image.",
	],
	"Descriptive (Casual)": [
		"Write a descriptive caption for this image in a casual tone.",
		"Write a descriptive caption for this image in a casual tone within {word_count} words.",
		"Write a {length} descriptive caption for this image in a casual tone.",
	],
	"Straightforward": [
		"Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
		"Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
		"Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
	],
	"Stable Diffusion Prompt": [
		"Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
		"Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
		"Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
	],
	"MidJourney": [
		"Write a MidJourney prompt for this image.",
		"Write a MidJourney prompt for this image within {word_count} words.",
		"Write a {length} MidJourney prompt for this image.",
	],
	"Danbooru tag list": [
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
		"Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
	],
	"e621 tag list": [
		"Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
		"Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
		"Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
	],
	"Rule34 tag list": [
		"Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
		"Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
		"Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
	],
	"Booru-like tag list": [
		"Write a list of Booru-like tags for this image.",
		"Write a list of Booru-like tags for this image within {word_count} words.",
		"Write a {length} list of Booru-like tags for this image.",
	],
	"Art Critic": [
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
	],
	"Product Listing": [
		"Write a caption for this image as though it were a product listing.",
		"Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
		"Write a {length} caption for this image as though it were a product listing.",
	],
	"Social Media Post": [
		"Write a caption for this image as if it were being used for a social media post.",
		"Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
		"Write a {length} caption for this image as if it were being used for a social media post.",
	],
        "Custom Photo": [
                """Write a detailed description for this image in 70 words or less. Include information about lighting. Include information about camera angle. If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc. Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry. Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot. Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.). Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…", \"You are looking at...\", etc.""",
                """Write a detailed description for this image in 70 words or less. Include information about lighting. Include information about camera angle. If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc. Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry. Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot. Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.). Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…", \"You are looking at...\", etc.""",
                """Write a detailed description for this image in 70 words or less. Include information about lighting. Include information about camera angle. If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc. Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry. Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot. Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.). Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…", \"You are looking at...\", etc.""",
        ],
}
NAME_OPTION = "If there is a person/character in the image you must refer to them as {name}."




g_processor = None
g_model: LlavaForConditionalGeneration | None = None
g_quant: str | None = None

def format_error(message: str) -> str:
	"""Format an error message for display in the UI."""
	return f'<div class="error-box">❌ Error: {message}</div>'

def format_info(message: str) -> str:
	"""Format an info message for display in the UI."""
	return f'<div class="info-box">ℹ️ {message}</div>'

def format_success(message: str) -> str:
	"""Format a success message for display in the UI."""
	return f'<div class="success-box">✅ {message}</div>'

def show_global_error(message: str):
	return gr.update(value=format_error(message), visible=True)

def hide_global_error():
	return gr.update(value="", visible=False)

def load_model(quant: str, status: gr.HTML | None = None):
	"""Load the model and processor if not already loaded."""
	global g_processor, g_model, g_quant
	if g_processor is None:
		print("Loading processor...")
		if status is not None:
			yield {status: format_info("Loading processor...")}
		
		try:
			g_processor = AutoProcessor.from_pretrained(MODEL_PATH)
			if g_processor.tokenizer.pad_token is None:
				g_processor.tokenizer.pad_token = g_processor.tokenizer.eos_token
		except Exception as e:
			error_msg = f"Failed to load processor: {e}"
			print(error_msg)
			if status is not None:
				yield {status: format_error(error_msg)}
			yield {global_error: show_global_error("Critical error: Model processor could not be loaded")}
			return
	
	if g_model is None or g_quant != quant:
		print("Loading model...")
		if status is not None:
			yield {status: format_info("Loading model weights...")}
		
		if g_model is not None:
			g_model = None
		
		gc.collect()
		torch.cuda.empty_cache()

		try:
			if quant == "bf16":
				g_model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype="bfloat16", device_map=0)
				assert isinstance(g_model, LlavaForConditionalGeneration), f"Expected LlavaForConditionalGeneration, got {type(g_model)}"
				# apply_liger_kernel_to_llama(model=g_model.language_model)  # Meow
			else:
				from transformers import BitsAndBytesConfig
				if quant == "8bit":
					qnt_config = BitsAndBytesConfig(
						load_in_8bit=True,
						llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],   # Transformer's Siglip implementation has bugs when quantized, so skip those.
					)
				elif quant == "nf4":
					qnt_config = BitsAndBytesConfig(
						load_in_4bit=True,
						bnb_4bit_quant_type="nf4",
						bnb_4bit_compute_dtype=torch.bfloat16,
						bnb_4bit_use_double_quant=True,
						llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],   # Transformer's Siglip implementation has bugs when quantized, so skip those.
					)
				else:
					raise ValueError(f"Unknown quantization type: {quant}")
				
				g_model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype="auto", device_map=0, quantization_config=qnt_config)
				assert isinstance(g_model, LlavaForConditionalGeneration), f"Expected LlavaForConditionalGeneration, got {type(g_model)}"

			g_model.eval()
			g_quant = quant
			# Hide any global error when model loads successfully
			yield {global_error: hide_global_error()}
		except Exception as e:
			error_msg = f"Failed to load model: {e}"
			print(error_msg)
			if status is not None:
				yield {status: format_error(error_msg)}
			yield {global_error: show_global_error(f"Critical error: Model failed to load with {quant} precision")}
			return
	
	if status is not None:
		yield {status: format_success("Model ready!")}


def build_prompt(caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str) -> str:
	# Choose the right template row in CAPTION_TYPE_MAP
	if caption_length == "any":
		map_idx = 0
	elif isinstance(caption_length, str) and caption_length.isdigit():
		map_idx = 1  # numeric-word-count template
	else:
		map_idx = 2  # length descriptor template
	
	prompt = CAPTION_TYPE_MAP[caption_type][map_idx]

	if extra_options:
		prompt += " " + " ".join(extra_options)
	
	return prompt.format(
		name=name_input or "{NAME}",
		length=caption_length,
		word_count=caption_length,
	)


def toggle_name_box(selected_options: list[str]):
	"""Show the name textbox only when the specific option is selected."""
	return gr.update(visible=NAME_OPTION in selected_options)


def _version(pkg_name: str):
	try:
		return metadata.version(pkg_name)
	except metadata.PackageNotFoundError:
		return "-- not installed --"


def print_system_info():
	# ---------- core library versions ----------
	lines = [
		f"Python            : {platform.python_version()} ({sys.executable})",
		f"PyTorch           : {torch.__version__}",
		f" ‣   CUDA build   : {torch.version.cuda or 'CPU-only build'}", # type: ignore
		f"transformers      : {_version('transformers')}",
		f"bitsandbytes      : {_version('bitsandbytes')}",
		f"liger_kernel      : {_version('liger_kernel')}",
	]

	# ---------- GPU information ----------
	if torch.cuda.is_available():
		gpu_lines = [f"GPUs (total {torch.cuda.device_count()}):"]
		for idx in range(torch.cuda.device_count()):
			cap_major, cap_minor = torch.cuda.get_device_capability(idx)
			props = torch.cuda.get_device_properties(idx)
			mem_gb = props.total_memory / (1024 ** 3)
			gpu_lines.append(
				f"  • [{idx}] {props.name} | "
				f"compute {cap_major}.{cap_minor} | "
				f"{mem_gb:.1f} GiB"
			)
		lines.extend(gpu_lines)
	else:
		lines.append("GPUs             : -- none detected / CUDA unavailable --")

	# ---------- pretty print ----------
	header = "\n🛠️  System configuration:\n"
	print(header + indent("\n".join(lines), "   "))


@torch.no_grad()
def chat_joycaption(input_image: Image.Image, prompt: str, temperature: float, top_p: float, max_new_tokens: int, quant: str) -> Generator[dict, None, None]:
	# Hide any previous global errors
	yield {global_error: hide_global_error()}

	if input_image is None:
		yield {single_status_output: format_error("No image selected for captioning. Please upload an image."), output_caption_single: None}
		return
	
	yield from load_model(quant=quant, status=single_status_output)
	gc.collect()
	torch.cuda.empty_cache()

	if g_model is None or g_processor is None:
		return
	
	yield {single_status_output: format_info("Generating caption...")}

	convo = [
		{
			"role": "system",
			# Beta One supports a wider range of system prompts, but this is a good default
			"content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.",
		},
		{
			"role": "user",
			"content": prompt.strip(),
		},
	]

	try:
		# Format the conversation
		# WARNING: HF's handling of chat's on Llava models is very fragile.  This specific combination of processor.apply_chat_template(), and processor() works
		# but if using other combinations always inspect the final input_ids to ensure they are correct.  Often times you will end up with multiple <bos> tokens
		# if not careful, which can make the model perform poorly.
		convo_string = g_processor.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
		assert isinstance(convo_string, str)

		# Process the inputs
		inputs = g_processor(text=[convo_string], images=[input_image], return_tensors="pt").to('cuda')
		inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

		streamer = TextIteratorStreamer(g_processor.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

		generate_kwargs = dict(
			**inputs,
			max_new_tokens=max_new_tokens,
			do_sample=True if temperature > 0 else False,
			suppress_tokens=None,
			use_cache=True,
			temperature=temperature if temperature > 0 else None,
			top_k=None,
			top_p=top_p if temperature > 0 else None,
			streamer=streamer,
		)

		t = Thread(target=g_model.generate, kwargs=generate_kwargs)
		t.start()

		outputs = []
		for text in streamer:
			outputs.append(text)
			yield {output_caption_single: "".join(outputs), single_status_output: format_info("Generating caption...")}
		
		t.join()
		yield {single_status_output: gr.update(value="Captioning complete!")}
	except Exception as e:
		error_msg = f"Error during generation: {str(e)}"
		print(error_msg)
		traceback.print_exc()
		yield {single_status_output: format_error(error_msg)}
		if "CUDA out of memory" in str(e):
			yield {global_error: show_global_error("CUDA out of memory error. Try using a more memory-efficient quantization setting (8bit or nf4).")}
		else:
			yield {global_error: show_global_error("Generation failed. See details below.")}


@torch.inference_mode()
def process_batch_files(
	files_list: list[NamedString],
	caption_type: str,
	caption_length: str | int,
	extra_options: list[str],
	name_input: str,
	temperature: float,
	top_p: float,
	max_new_tokens: int,
	num_workers: int,
	batch_size: int,
	quant: str,
	progress = gr.Progress(track_tqdm=True),
) -> Generator[dict, None, None]:
	# Hide any previous global errors
	yield {global_error: hide_global_error()}

	if not files_list:
		yield {batch_status_output: format_error("No files selected for batch processing. Please upload one or more image files."), batch_zip_output: None}
		return
	
	yield from load_model(quant=quant, status=batch_status_output)
	gc.collect()
	torch.cuda.empty_cache()

	if g_model is None or g_processor is None:
		return
	
	try:
		vision_dtype = g_model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
		vision_device = g_model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
		language_device = g_model.language_model.get_input_embeddings().weight.device

		captions_dict: dict[str, str] = {}
		prompt = build_prompt(caption_type, caption_length, extra_options, name_input)
		system_prompt = "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions."
		tasks = [(Path(f), system_prompt, prompt) for f in files_list]
		dataloader = torch.utils.data.DataLoader(ImageDataset(tasks), num_workers=num_workers, shuffle=False, drop_last=False, batch_size=batch_size, collate_fn = partial(collate_fn, processor=g_processor))
		missing_paths = set(Path(f) for f in files_list)

		yield {batch_status_output: format_info(f"Processing {len(files_list)} images...")}

		with tqdm(total=len(files_list), desc="Processing", unit="image") as pbar:
			for batch in dataloader:
				if len(batch['paths']) == 0:
					continue
		
				# Move to GPU
				pixel_values = batch['pixel_values'].to(vision_device, non_blocking=True)
				input_ids = batch['input_ids'].to(language_device, non_blocking=True)
				attention_mask = batch['attention_mask'].to(language_device, non_blocking=True)

				# Generate the captions
				generate_ids = g_model.generate(
					input_ids=input_ids,
					pixel_values=pixel_values.to(vision_dtype),
					attention_mask=attention_mask,
					max_new_tokens=max_new_tokens,
					do_sample=True if temperature > 0 else False,
					suppress_tokens=None,
					use_cache=True,
					temperature=temperature if temperature > 0 else None,
					top_k=None,
					top_p=top_p if temperature > 0 else None,
				)

				# Trim off the prompts and decode
				preds = generate_ids[:, input_ids.shape[1]:]
				captions = g_processor.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)

				for path, caption in zip(batch['paths'], captions):
					caption_filename = path.with_suffix(".txt").name
					if caption_filename in captions_dict:
						print(f"Warning: Duplicate caption filename {caption_filename} for {path}. Skipping.")
						continue
					captions_dict[caption_filename] = caption.strip()
					missing_paths.discard(path)
				
				pbar.update(len(batch['paths']))
				yield {batch_status_output: format_info(f"Processed {pbar.n}/{len(files_list)} images...")}
		
		# Build ZIP
		tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
		with open(tmp.name, "wb") as f:
			with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as zf:
				for fname, text in captions_dict.items():
					zf.writestr(fname, text)
		
		if len(missing_paths) > 0:
			error_msg = f"Warning: {len(missing_paths)} images could not be processed. Check the console for details."
			print(error_msg)
			yield {batch_status_output: format_error(error_msg), batch_zip_output: tmp.name}
		else:
			yield {batch_status_output: format_success(f"Batch processing complete! Processed {len(captions_dict)} images."), batch_zip_output: tmp.name}

	except Exception as e:
		error_msg = f"Error during batch processing: {str(e)}"
		print(error_msg)
		traceback.print_exc()
		yield {batch_status_output: format_error(error_msg), batch_zip_output: None}
		if "CUDA out of memory" in str(e):
			yield {global_error: show_global_error("CUDA out of memory! Try reducing batch size or using a more memory-efficient quantization.")}
		else:
			yield {global_error: show_global_error("Batch processing failed. See details below.")}


class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, tasks: list[tuple[Path, str, str]]):
		self.tasks = tasks
	
	def __len__(self):
		return len(self.tasks)
	
	def __getitem__(self, idx: int) -> tuple[Path, Image.Image, str, str] | None:
		path, system_prompt, user_prompt = self.tasks[idx]
		try:
			image = Image.open(path).convert("RGB")
		except Exception as e:
			print(f"Error processing image {path}: {e}")
			traceback.print_exc()
			return None

		return path, image, system_prompt, user_prompt


def collate_fn(batch: list[tuple[Path, Image.Image, str, str] | None], *, processor):
	convos = []
	paths = []
	images = []

	for path, image, system_prompt, user_prompt in (b for b in batch if b is not None):
		convos.append([
			{"role": "system", "content": system_prompt.strip()},
			{"role": "user", "content": user_prompt.strip()},
		])
		images.append(image)
		paths.append(path)
	
	if len(images) == 0:
		return { "paths": [] }
	
	convo_strings = processor.apply_chat_template(convos, tokenize = False, add_generation_prompt = True)
	inputs = processor(text=convo_strings, images=images, return_tensors="pt", padding="longest")
	inputs['paths'] = paths

	return inputs


with gr.Blocks() as demo:
	gr.HTML(TITLE)

	global_error = gr.HTML(visible=False)

	with gr.Row(equal_height=True) as top_row:
		with gr.Column(scale=1):
			model_quantization = gr.Dropdown(
				choices=["bf16", "int8", "nf4"],
				value="bf16",
				label="Model Quantization",
				info="Model quantization level (bf16=highest quality, 8bit/nf4=lower memory)",
			)
		
		with gr.Column(scale=3):
			with gr.Row():
                                caption_type = gr.Dropdown(
                                        choices=list(CAPTION_TYPE_MAP.keys()),
                                        value="Straightforward",
                                        label="Caption Type",
                                )

                                caption_length = gr.Dropdown(
                                        choices=["any", "very short", "short", "medium-length", "long", "very long"] +
                                                        [str(i) for i in range(20, 261, 10)],
                                        label="Caption Length",
                                        value="70",
                                )

	with gr.Accordion("Extra Options", open=False):
		extra_options = gr.CheckboxGroup(
			choices=[
				NAME_OPTION,
				"Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
				"Include information about lighting.",
				"Include information about camera angle.",
				"Include information about whether there is a watermark or not.",
				"Include information about whether there are JPEG artifacts or not.",
				"If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
				"Do NOT include anything sexual; keep it PG.",
				"Do NOT mention the image's resolution.",
				"You MUST include information about the subjective aesthetic quality of the image from low to very high.",
				"Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
				"Do NOT mention any text that is in the image.",
				"Specify the depth of field and whether the background is in focus or blurred.",
				"If applicable, mention the likely use of artificial or natural lighting sources.",
				"Do NOT use any ambiguous language.",
				"Include whether the image is sfw, suggestive, or nsfw.",
				"ONLY describe the most important elements of the image.",
				"If it is a work of art, do not include the artist's name or the title of the work.",
				"Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",
				"""Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.""",
				"Do NOT use polite euphemisms—lean into blunt, casual phrasing.",
				"Include information about the ages of any people/characters when applicable.",
				"Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
				"Do not mention the mood/feeling/etc of the image.",
				"Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.).",
				"If there is a watermark, you must mention it.",
				"""Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…”, "You are looking at...", etc.""",
			],
			label="Select one or more",
		)
		
	name_input = gr.Textbox(label="Person / Character Name", visible=False)

	with gr.Accordion("Generation settings", open=False):
		with gr.Row():
			with gr.Column(scale=1):
				temperature_slider = gr.Slider(
					minimum=0.0, maximum=2.0, value=0.6, step=0.05,
					label="Temperature",
					info="Higher values make the output more random, lower values make it more deterministic."
				)
			with gr.Column(scale=1):
				top_p_slider = gr.Slider(
					minimum=0.0, maximum=1.0, value=0.9, step=0.01,
					label="Top-p"
				)
			with gr.Column(scale=1):
				max_tokens_slider = gr.Slider(
					minimum=1, maximum=2048, value=512, step=1,
					label="Max New Tokens",
					info="Maximum number of tokens to generate.  The model will stop generating if it reaches this limit."
				)
	
	gr.HTML("<hr>")

	with gr.Tabs() as tabs:
		# Single Image Tab
		with gr.TabItem("Single Image Processing", id="single_tab"):
			gr.HTML('<div class="panel-header"><h3>Process a Single Image</h3></div>')

			single_status_output = gr.HTML("")

			with gr.Row():
				with gr.Column(scale=1):
					input_image_single = gr.Image(type="pil", label="Upload Image", height=400, elem_id="single_image_input")
				
				with gr.Column(scale=1):
					initial_single_prompt = build_prompt(caption_type.value, caption_length.value, extra_options.value, name_input.value)
					prompt_box_single = gr.Textbox(lines=4, label="Confirm or Edit Prompt", value=initial_single_prompt, interactive=True, elem_id="single_prompt_box")
					run_button_single = gr.Button("Caption", variant="primary")
					output_caption_single = gr.Textbox(label="Generated Caption", lines=8, interactive=True, elem_id="single_output_box")
		
		# Batch Processing Tab
		with gr.TabItem("Batch Processing", id="batch_tab"):
			gr.HTML('<div class="panel-header"><h3>Process Multiple Images</h3><p>Upload multiple images to generate captions in bulk. You\'ll receive a ZIP file with all the captions.</p></div>')

			batch_status_output = gr.HTML("")

			with gr.Row():
				with gr.Column(scale=3):
					input_files_batch = gr.File(file_count="multiple", file_types=None, label="Upload Images (Batch)", elem_id="batch_file_input")
				
				with gr.Column(scale=2):
					with gr.Group():
						gr.HTML("<h4>Batch Processing Settings</h4>")

						with gr.Row():
							num_workers_slider = gr.Slider(
								minimum=0, maximum=32, step=1, value=4,
								label="DataLoader Workers",
								info="CPU worker processes (0 = no multiprocessing)"
							)
						
						with gr.Row():
							batch_size_slider = gr.Slider(
								minimum=1, maximum=32, step=1, value=4,
								label="Batch Size",
								info="Images to process at once (affects vram usage)"
							)

						run_button_batch = gr.Button("Start Batch Process & Create ZIP", variant="primary")
				
			with gr.Row():
				batch_zip_output = gr.File(label="Download captions.zip", elem_id="batch_zip_output")

			
	
	# Documentation section
	with gr.Accordion("Documentation", open=False):
		gr.HTML(DESCRIPTION)
	
	# Wire up events

	# Show name input box only when the name option is selected
	extra_options.change(toggle_name_box, inputs=[extra_options], outputs=[name_input])

	# Update the prompt box when any of the options change
	for ctrl in (caption_type, caption_length, extra_options, name_input):
		ctrl.change(
			build_prompt,
			inputs=[caption_type, caption_length, extra_options, name_input],
			outputs=prompt_box_single,
		)
	
	# Handle single image captioning
	run_button_single.click(
		chat_joycaption,
		inputs=[input_image_single, prompt_box_single, temperature_slider, top_p_slider, max_tokens_slider, model_quantization],
		outputs=[single_status_output, output_caption_single, global_error],
	)

	# Handle batch processing
	run_button_batch.click(
		process_batch_files,
		inputs=[input_files_batch, caption_type, caption_length, extra_options, name_input, temperature_slider, top_p_slider, max_tokens_slider, num_workers_slider, batch_size_slider, model_quantization],
		outputs=[batch_status_output, batch_zip_output, global_error],
	)


if __name__ == "__main__":
	print_system_info()
	demo.launch()
