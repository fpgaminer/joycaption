# JoyCaption - Advanced Image Captioning Tool

![JoyCaption Screenshot](gradio-app.webp)

This is a simple Gradio GUI for the JoyCaption model.

## Installation

### Prerequisites

- Python 3.8+ 
- CUDA-capable GPU (recommended)
- At least 24GB VRAM for bf16 precision (8GB for nf4 quantized mode)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/joy-caption.git
   cd joy-caption/gradio-app
   ```

2. Set up a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your web browser and navigate to the URL displayed in the terminal (typically http://127.0.0.1:7860)

## Usage

### Single Image Captioning

1. Upload an image using the left-hand panel
2. Select a caption type and desired caption length
3. (Optional) Open "Extra Options" and select any additional parameters
4. (Optional) Adjust generation settings like temperature and top-p values
5. Click "Caption" to generate
6. The generated caption will appear in the output box and can be copied or edited

### Batch Processing

1. Switch to the "Batch Processing" tab
2. Upload multiple images (PNG/JPEG/WEBP)
3. Set the DataLoader Workers (CPU processes) and Batch Size based on your system capabilities
4. Click "Start Batch Process & Create ZIP"
5. Download the ZIP file containing all captions when processing completes

## Caption Types

| Mode | Description |
|------|-------------|
| **Descriptive** | Formal, detailed prose description |
| **Descriptive (Casual)** | Similar to Descriptive but with a friendlier, conversational tone |
| **Straightforward** | Objective, no fluff, and more succinct than Descriptive |
| **Stable Diffusion Prompt** | Reverse-engineers a prompt for Stable Diffusion models |
| **MidJourney** | Similar to the above but tuned for MidJourney's prompt style |
| **Danbooru tag list** | Comma-separated tags following Danbooru conventions |
| **e621 tag list** | Alphabetical, namespaced tags in e621 style |
| **Rule34 tag list** | Rule34 style alphabetical tag dump |
| **Booru-like tag list** | Looser tag list for general labeling |
| **Art Critic** | Art-historical commentary on composition, symbolism, style, etc. |
| **Product Listing** | Marketing copy as if selling the depicted object |
| **Social Media Post** | Catchy caption for social media platforms |

> **Note on Booru modes**: They're optimized for anime-style/illustration imagery; accuracy may decrease with real-world photographs or abstract artwork.

## Model Quantization

Select from different precision levels to balance quality and memory usage:

- **bf16**: Highest quality, highest VRAM usage
- **8-bit**: Good balance of quality and VRAM efficiency
- **nf4 (4-bit)**: Lowest VRAM usage with slight quality degradation
