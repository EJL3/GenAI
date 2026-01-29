# Cooking Assistant - Setup Instructions

This guide will walk you through setting up a local LLM-based cooking assistant for the workshop.

## Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional, but recommended for faster performance)
- At least 8GB of RAM

## Step 1: Create a Virtual Environment

A virtual environment keeps your project dependencies isolated from other Python projects.

```bash
# Create a new virtual environment named 'cooking_assistant_env'
python3 -m venv cooking_assistant_env

# Activate the virtual environment
# On Linux/Mac:
source cooking_assistant_env/bin/activate

# On Windows:
# cooking_assistant_env\Scripts\activate
```

## Step 2: Install Dependencies

Install the required Python packages for running the LLM:

```bash
# Upgrade pip to the latest version
pip install --upgrade pip

# Install PyTorch (with CUDA support for GPU acceleration)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only (if you don't have a GPU):
# pip install torch torchvision torchaudio

# Install transformers and accelerate for model loading and inference
pip install transformers accelerate

# Install bitsandbytes for 4-bit quantization (reduces memory usage)
# Note: bitsandbytes requires CUDA/GPU support
pip install bitsandbytes

# Optional: Install sentencepiece for tokenizer support
pip install sentencepiece protobuf
```

## Step 3: Run the Cooking Assistant

Once all dependencies are installed, you can run the assistant:

```bash
python cooking_assistant.py
```

The first time you run the script, it will download the model (~2-4GB). This may take a few minutes depending on your internet connection.

## Troubleshooting

### Issue: "CUDA not available" warning
- **Solution**: The script will automatically fall back to CPU. GPU is recommended but not required.

### Issue: Out of memory error
- **Solution**: The script uses 4-bit quantization to reduce memory usage. If you still encounter issues, try closing other applications or use a smaller model.

### Issue: bitsandbytes installation fails
- **Solution**: This library requires CUDA. If you're on CPU-only, you can skip 4-bit quantization by modifying the script (remove the `load_in_4bit` parameter).

## Models Used

This script supports two lightweight models suitable for local deployment:

1. **Qwen2.5-3B-Instruct** (~3 billion parameters)
   - Model ID: `Qwen/Qwen2.5-3B-Instruct`
   
2. **Phi-3.5-mini** (~3.8 billion parameters)
   - Model ID: `microsoft/Phi-3.5-mini-instruct`

You can switch between models by modifying the `MODEL_NAME` variable in the Python script.

## Workshop Notes

- The model runs entirely on your local machine - no API keys needed!
- The guardrail function demonstrates basic input validation
- The Chef persona shows how system prompts guide LLM behavior
