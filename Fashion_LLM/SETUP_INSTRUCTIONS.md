# Fashion Architect - Setup Instructions

This guide will walk you through setting up a text-to-image fashion design generator for the workshop.

## Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with at least 4GB VRAM (recommended for faster generation)
- At least 16GB of RAM
- ~10GB of disk space for models

## Step 1: Create a Virtual Environment

A virtual environment keeps your project dependencies isolated from other Python projects.

```bash
# Create a new virtual environment named 'fashion_env'
python3 -m venv fashion_env

# Activate the virtual environment
# On Linux/Mac:
source fashion_env/bin/activate

# On Windows:
# fashion_env\Scripts\activate
```

## Step 2: Install Dependencies

Install the required Python packages for running Stable Diffusion:

```bash
# Upgrade pip to the latest version
pip install --upgrade pip

# Install PyTorch (with CUDA support for GPU acceleration)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only (NOT recommended for image generation - very slow):
# pip install torch torchvision torchaudio

# Install diffusers library (for Stable Diffusion pipeline)
pip install diffusers

# Install transformers (for text encoding)
pip install transformers

# Install accelerate (for optimized inference)
pip install accelerate

# Install additional image processing libraries
pip install Pillow
```

## Step 3: Hugging Face Authentication (Optional but Recommended)

Some Stable Diffusion models may require Hugging Face authentication:

```bash
# Install huggingface-hub
pip install huggingface-hub

# Login to Hugging Face (you'll need a free account at huggingface.co)
huggingface-cli login
```

You can create a free account at [https://huggingface.co](https://huggingface.co) and generate an access token from your settings.

## Step 4: Run the Fashion Architect

Once all dependencies are installed, you can run the designer:

```bash
python fashion_architect.py
```

The first time you run the script, it will download the Stable Diffusion v1.5 model (~4-5GB). This may take several minutes depending on your internet connection.

## Troubleshooting

### Issue: "CUDA out of memory" error
- **Solution**: The script will attempt to optimize memory usage. If issues persist:
  - Close other GPU-intensive applications
  - Reduce the image resolution in the script (modify `height` and `width` parameters)
  - Enable `enable_attention_slicing()` in the script (already included)

### Issue: "No module named 'diffusers'" error
- **Solution**: Make sure you activated the virtual environment and installed all dependencies:
  ```bash
  source fashion_env/bin/activate  # or fashion_env\Scripts\activate on Windows
  pip install diffusers transformers accelerate
  ```

### Issue: Very slow generation on CPU
- **Solution**: Image generation on CPU can take 5-10 minutes per image. GPU is highly recommended.

### Issue: Low-quality or artifacts in generated images
- **Solution**: 
  - Increase the number of inference steps (modify `num_inference_steps` in script)
  - Adjust the guidance scale (higher = more adherence to prompt)
  - Refine your negative prompt to filter specific artifacts

## Model Information

This script uses **Stable Diffusion v1.5**:
- Model ID: `runwayml/stable-diffusion-v1-5`
- Resolution: 512x512 pixels (default)
- Inference steps: 50 (configurable)
- Guidance scale: 7.5 (configurable)

## Workshop Notes

- The model generates images locally on your machine
- Each generation takes ~5-10 seconds on a modern GPU
- Experiment with different combinations of garment types, materials, and styles
- The negative prompt is crucial for quality - it tells the model what NOT to include
- Save your favorite prompts for reproducible results!

## Performance Tips

1. **GPU Memory**: If running out of memory, use `pipe.enable_attention_slicing()`
2. **Speed**: Lower `num_inference_steps` for faster (but lower quality) generation
3. **Quality**: Higher `guidance_scale` (7.5-15) makes images follow prompts more closely
4. **Consistency**: Use the same `seed` value for reproducible results
