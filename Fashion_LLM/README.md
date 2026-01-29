# ✨ Fashion Architect - AI-Powered Fashion Design Generator

An educational text-to-image pipeline built for workshops using Stable Diffusion v1.5. This project teaches you how to generate professional fashion designs from structured text prompts.

## 📋 Overview

This workshop project demonstrates:
- Setting up a Stable Diffusion pipeline using the `diffusers` library
- Creating structured prompts from modular inputs
- Using negative prompts to filter artifacts and improve quality
- Saving generated images with organized naming
- Optimizing for GPU/CPU performance

## 🎯 Features

- **Modular Input System**: Three-part input (Garment Type, Material, Aesthetic)
- **Structured Prompts**: Automatically combines inputs into effective prompts
- **Quality Control**: Built-in negative prompts filter common artifacts
- **Professional Output**: Generates 512x512 fashion photography
- **Timestamped Saves**: Automatic file naming with timestamps
- **Memory Optimized**: Attention slicing for lower VRAM usage
- **Educational**: Extensive code comments for learning

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv fashion_env
source fashion_env/bin/activate  # Linux/Mac
# OR: fashion_env\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

See [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) for detailed installation steps.

### 2. Run the Fashion Architect

```bash
python fashion_architect.py
```

The first run will download Stable Diffusion v1.5 (~4-5GB).

## 💡 Example Usage

```
✨ FASHION ARCHITECT - DESIGN YOUR CREATION

👗 Step 1: Choose a Garment Type
   Enter garment type: evening gown

🧵 Step 2: Select Material or Texture
   Enter material/texture: velvet

✨ Step 3: Pick an Aesthetic Style
   Enter aesthetic style: baroque

🎨 Generating design...
✅ Design generated successfully!
💾 Saved to: generated_designs/fashion_design_20260129_143052.png
```

This generates a high-quality image of a baroque evening gown made of velvet.

## 🎨 Example Combinations

1. **Cyberpunk Streetwear**
   - Garment: leather jacket
   - Material: distressed leather
   - Style: cyberpunk

2. **Minimalist Elegance**
   - Garment: flowing dress
   - Material: silk
   - Style: minimalist

3. **Art Deco Luxury**
   - Garment: tailored suit
   - Material: wool
   - Style: art deco

4. **Futuristic Fusion**
   - Garment: kimono
   - Material: metallic fabric
   - Style: futuristic

5. **Gothic Romance**
   - Garment: corset dress
   - Material: lace
   - Style: gothic

## 🛠️ How It Works

### 1. Modular Input Collection
```python
garment_type = "evening gown"
material = "velvet"
aesthetic = "baroque"
```

### 2. Structured Prompt Building
The `build_fashion_prompt()` function combines inputs:
```python
prompt = "A high-fashion photograph of a baroque evening gown made of velvet, 
          professional photography, studio lighting, detailed texture, 
          8k resolution, fashion magazine quality..."
```

### 3. Negative Prompt Filtering
```python
negative_prompt = "deformed, blurry, low quality, bad anatomy, 
                   watermark, text..."
```

### 4. Image Generation
```python
image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=7.5
)
```

### 5. Save Output
```python
image.save("generated_designs/fashion_design.png")
```

## 📊 Configuration Parameters

You can customize these in the script:

```python
IMAGE_WIDTH = 512              # Image width (512 optimal for SD v1.5)
IMAGE_HEIGHT = 512             # Image height
NUM_INFERENCE_STEPS = 50       # Quality vs speed (20-100)
GUIDANCE_SCALE = 7.5           # Prompt adherence (7-15)
```

### Parameter Guide

- **Inference Steps**: 
  - 20 = Fast, lower quality (~3-5 seconds)
  - 50 = Balanced (recommended)
  - 100 = Highest quality, slower (~20-30 seconds)

- **Guidance Scale**:
  - 5-7 = More creative, less literal
  - 7.5-10 = Balanced (recommended)
  - 10-15 = Very literal, follows prompt closely

## 🎓 Educational Features

### Modular Code Structure

1. **Configuration Section**: All settings in one place
2. **Prompt Builder**: Separated function for clarity
3. **Pipeline Setup**: Reusable initialization
4. **Generation Function**: Core logic isolated
5. **Input Handlers**: User interaction separated

### Key Concepts Taught

- **Text-to-Image Diffusion**: How prompts become images
- **Negative Prompts**: Quality control technique
- **Structured Prompting**: Breaking complex prompts into parts
- **Pipeline Optimization**: Memory and speed techniques
- **Modular Design**: Organizing code for reusability

## 📈 Performance Tips

### For Best Quality
```python
NUM_INFERENCE_STEPS = 100
GUIDANCE_SCALE = 10
```

### For Speed (Lower Quality)
```python
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5
```

### For Reproducibility
```python
# Set a fixed seed in generate_fashion_design()
seed = 42
```

## 🔧 Customization Ideas

1. **Add More Style Presets**: Create preset combinations for popular styles
2. **Batch Generation**: Generate multiple variations at once
3. **Image Upscaling**: Add post-processing for higher resolution
4. **Style Mixing**: Blend multiple aesthetics
5. **Color Palette**: Add color specification inputs
6. **Reference Images**: Use img2img for style transfer

## 📊 System Requirements

- **Minimum**: 16GB RAM, NVIDIA GPU with 6GB VRAM
- **Recommended**: 32GB RAM, NVIDIA GPU with 8GB+ VRAM
- **CPU-Only**: Possible but very slow (5-10 minutes per image)
- **Disk Space**: ~15GB (model + dependencies)

## 🐛 Troubleshooting

**Out of VRAM?**
- Already using `enable_attention_slicing()`
- Reduce `IMAGE_WIDTH` and `IMAGE_HEIGHT` to 384 or 256
- Close other GPU applications

**Low Quality Images?**
- Increase `NUM_INFERENCE_STEPS` to 75-100
- Adjust `GUIDANCE_SCALE` to 10-12
- Refine your prompts with more descriptive words
- Check negative prompt includes your specific issues

**Generation Too Slow?**
- Reduce `NUM_INFERENCE_STEPS` to 25-30
- Use GPU instead of CPU
- Consider using a smaller model

## 📖 Learning Resources

- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Stable Diffusion Guide](https://stability.ai/stable-diffusion)
- [Prompt Engineering for Images](https://prompthero.com/stable-diffusion-prompt-guide)
- [Negative Prompts Guide](https://stable-diffusion-art.com/negative-prompt/)

## 🎨 Gallery

Generated images are saved to `generated_designs/` with timestamps:
```
generated_designs/
├── fashion_design_20260129_143052.png
├── fashion_design_20260129_143215.png
└── fashion_design_20260129_143341.png
```

## 📝 License

Educational use - free to modify and share!

## 🙏 Acknowledgments

- Built with Hugging Face Diffusers
- Uses Stable Diffusion v1.5 by Runway ML
- Created for educational workshops

---

**Happy Designing with AI! ✨👗**
