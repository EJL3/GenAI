import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

# Stable Diffusion model to use
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Output settings
OUTPUT_DIR = "generated_designs"
DEFAULT_OUTPUT_FILENAME = "fashion_design.png"

# Generation parameters
IMAGE_WIDTH = 512  # Width in pixels (512 is optimal for SD v1.5)
IMAGE_HEIGHT = 512  # Height in pixels
NUM_INFERENCE_STEPS = 50  # More steps = higher quality but slower (20-100 typical)
GUIDANCE_SCALE = 7.5  # How closely to follow the prompt (7-15 typical)

# Negative prompt - what we DON'T want in the image
# This is crucial for filtering out common artifacts and unwanted elements
NEGATIVE_PROMPT = """
deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy,
extra limb, missing limb, floating limbs, mutated hands and fingers,
disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation,
low quality, worst quality, low resolution, pixelated, jpeg artifacts,
watermark, text, signature, username, grainy, out of focus
"""


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

def build_fashion_prompt(garment_type: str, material: str, aesthetic: str) -> str:
    """
    Construct a structured prompt from user inputs.
    
    This function combines the three key elements of fashion design into
    a coherent prompt that Stable Diffusion can understand and generate.
    
    Args:
        garment_type (str): Type of clothing (e.g., "dress", "jacket", "pants")
        material (str): Fabric or texture (e.g., "silk", "leather", "denim")
        aesthetic (str): Design style (e.g., "cyberpunk", "minimalist", "baroque")
        
    Returns:
        str: Formatted prompt for image generation
        
    Example:
        >>> prompt = build_fashion_prompt("evening gown", "velvet", "baroque")
        >>> print(prompt)
        A high-fashion photograph of a baroque evening gown made of velvet, ...
    """
    # Base template for professional fashion photography
    base_template = "A high-fashion photograph of a {aesthetic} {garment} made of {material}"
    
    # Quality enhancers - these keywords improve image generation
    quality_keywords = [
        "professional photography",
        "studio lighting",
        "detailed texture",
        "8k resolution",
        "fashion magazine quality",
        "runway presentation",
        "elegant design",
        "haute couture"
    ]
    
    # Combine the base prompt with quality enhancers
    prompt = base_template.format(
        aesthetic=aesthetic.lower(),
        garment=garment_type.lower(),
        material=material.lower()
    )
    
    # Add quality keywords to enhance generation
    full_prompt = f"{prompt}, {', '.join(quality_keywords)}"
    
    return full_prompt


# =============================================================================
# PIPELINE SETUP
# =============================================================================

def setup_pipeline():
    """
    Initialize the Stable Diffusion pipeline with optimizations.
    
    This function:
    1. Detects available hardware (GPU or CPU)
    2. Loads the Stable Diffusion v1.5 model
    3. Configures an efficient scheduler (DPM-Solver++)
    4. Applies memory optimizations
    
    Returns:
        StableDiffusionPipeline: Configured pipeline ready for generation
    """
    print("🎨 Initializing Fashion Architect...")
    print(f"📦 Loading Stable Diffusion model: {MODEL_ID}")
    
    # Detect available device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16  # Use half precision for GPU (saves memory)
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        dtype = torch.float32  # CPU requires full precision
        print("🖥️  Using CPU (Warning: Generation will be very slow)")
    
    # Load the Stable Diffusion pipeline
    print("🧠 Loading model (this may take a few minutes on first run)...")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,  # Disable safety checker for workshop use
    )
    
    # Use DPM-Solver++ scheduler for faster, high-quality generation
    # This scheduler produces good results in fewer steps than the default
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    
    # Move pipeline to the appropriate device
    pipeline = pipeline.to(device)
    
    # Memory optimization: Attention slicing reduces VRAM usage
    # This allows generation on GPUs with less memory
    pipeline.enable_attention_slicing()
    
    print("✅ Pipeline ready!\n")
    
    return pipeline


# =============================================================================
# IMAGE GENERATION
# =============================================================================

def generate_fashion_design(
    pipeline,
    garment_type: str,
    material: str,
    aesthetic: str,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    seed: int = None
) -> Image.Image:
    """
    Generate a fashion design image based on user inputs.
    
    This is the main generation function that:
    1. Builds the prompt from inputs
    2. Runs the diffusion process
    3. Saves the generated image
    
    Args:
        pipeline: The Stable Diffusion pipeline
        garment_type (str): Type of clothing
        material (str): Fabric or texture
        aesthetic (str): Design style
        output_filename (str): Name for the saved image file
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        PIL.Image: The generated image
    """
    # Build the structured prompt
    prompt = build_fashion_prompt(garment_type, material, aesthetic)
    
    print("=" * 70)
    print("🎨 FASHION ARCHITECT - DESIGN GENERATION")
    print("=" * 70)
    print(f"\n📝 Garment Type: {garment_type}")
    print(f"🧵 Material: {material}")
    print(f"✨ Aesthetic: {aesthetic}")
    print(f"\n💡 Generated Prompt:")
    print(f"   {prompt}")
    print(f"\n🚫 Negative Prompt:")
    print(f"   {NEGATIVE_PROMPT.strip()}")
    print(f"\n⚙️  Settings:")
    print(f"   - Resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"   - Inference Steps: {NUM_INFERENCE_STEPS}")
    print(f"   - Guidance Scale: {GUIDANCE_SCALE}")
    if seed:
        print(f"   - Seed: {seed}")
    print("\n🎨 Generating design (this may take 5-30 seconds)...\n")
    
    # Set random seed for reproducibility (optional)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    # Generate the image
    # This is where the magic happens - the diffusion model creates an image
    # from pure noise, guided by your prompt
    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT.strip(),
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            generator=generator
        )
    
    # Extract the generated image
    image = result.images[0]
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create full output path
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Save the generated image
    image.save(output_path)
    
    print(f"✅ Design generated successfully!")
    print(f"💾 Saved to: {output_path}\n")
    
    return image


# =============================================================================
# USER INPUT HANDLERS
# =============================================================================

def get_user_inputs() -> tuple[str, str, str]:
    """
    Collect fashion design parameters from the user.
    
    Returns:
        tuple: (garment_type, material, aesthetic)
    """
    print("=" * 70)
    print("🎨 FASHION ARCHITECT - DESIGN YOUR CREATION")
    print("=" * 70)
    print()
    
    # Garment Type suggestions
    print("👗 Step 1: Choose a Garment Type")
    print("   Examples: evening gown, leather jacket, flowing dress, tailored suit,")
    print("            cargo pants, crop top, trench coat, kimono, blazer, skirt")
    garment_type = input("\n   Enter garment type: ").strip()
    
    print()
    
    # Material/Texture suggestions
    print("🧵 Step 2: Select Material or Texture")
    print("   Examples: silk, velvet, leather, denim, lace, satin, cotton,")
    print("            wool, mesh, metallic fabric, sequins, chiffon, brocade")
    material = input("\n   Enter material/texture: ").strip()
    
    print()
    
    # Aesthetic Style suggestions
    print("✨ Step 3: Pick an Aesthetic Style")
    print("   Examples: cyberpunk, minimalist, baroque, futuristic, vintage,")
    print("            gothic, bohemian, art deco, streetwear, romantic, avant-garde")
    aesthetic = input("\n   Enter aesthetic style: ").strip()
    
    print()
    
    return garment_type, material, aesthetic


def display_examples():
    """
    Display example combinations to inspire users.
    """
    print("\n💡 Example Combinations:")
    print("   1. Garment: evening gown | Material: velvet | Style: baroque")
    print("   2. Garment: leather jacket | Material: distressed leather | Style: cyberpunk")
    print("   3. Garment: flowing dress | Material: silk | Style: minimalist")
    print("   4. Garment: tailored suit | Material: wool | Style: art deco")
    print("   5. Garment: kimono | Material: brocade | Style: futuristic")
    print()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main application loop for the Fashion Architect.
    """
    print("\n" + "=" * 70)
    print("✨ WELCOME TO FASHION ARCHITECT ✨")
    print("AI-Powered Fashion Design Generator")
    print("=" * 70)
    
    # Display some example combinations to inspire users
    display_examples()
    
    # Initialize the Stable Diffusion pipeline
    pipeline = setup_pipeline()
    
    # Main generation loop
    while True:
        # Get user inputs for the design
        garment_type, material, aesthetic = get_user_inputs()
        
        # Validate inputs
        if not garment_type or not material or not aesthetic:
            print("❌ Error: All three inputs are required!")
            continue
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"fashion_design_{timestamp}.png"
        
        try:
            # Generate the fashion design
            image = generate_fashion_design(
                pipeline=pipeline,
                garment_type=garment_type,
                material=material,
                aesthetic=aesthetic,
                output_filename=output_filename,
                seed=None  # Random seed for variety (set to a number for reproducibility)
            )
            
            print("=" * 70)
            print()
            
        except Exception as e:
            print(f"❌ Error generating design: {e}")
            print("Please try again with different inputs.\n")
        
        # Ask if user wants to generate another design
        another = input("Generate another design? (y/n): ").strip().lower()
        if another not in ['y', 'yes']:
            print("\n✨ Thank you for using Fashion Architect! Happy designing! 👗\n")
            break


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Entry point of the script.
    This block only runs when the script is executed directly,
    not when imported as a module.
    """
    try:
        main()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n\n✨ Fashion Architect interrupted. Goodbye! 👗\n")
    except Exception as e:
        # Catch any unexpected errors
        print(f"\n❌ An unexpected error occurred: {e}")
        print("Please check your installation and GPU/CUDA setup.\n")
