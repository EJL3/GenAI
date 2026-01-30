import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings

# Suppress some common warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================
# Model name from Hugging Face Model Hub
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"      

# Force device - "cpu" or "cuda"
FORCE_DEVICE = "cuda"  # Try GPU first with smaller model

# Non-cooking topics that should trigger the guardrail
BLOCKED_KEYWORDS = [
    'code', 'programming', 'python', 'javascript', 'html',
    'politics', 'political', 'election', 'government',
    'history', 'historical', 'war', 'battle',
    'math', 'mathematics', 'calculus', 'algebra',
    'medicine', 'medical', 'diagnosis', 'prescription',
    'legal', 'lawyer', 'lawsuit', 'court'
]

# The Chef's personality and role definition
CHEF_SYSTEM_PROMPT = """You are Chef Auguste, a passionate and knowledgeable culinary expert with 30 years of experience in French and Italian cuisine. You are enthusiastic, warm, and love to share your cooking wisdom with others.

Your role:
- Help users with cooking questions, recipes, techniques, and ingredient substitutions
- Provide clear, practical advice that home cooks can follow
- Share interesting culinary facts and tips
- Be encouraging and supportive of cooking experiments
- Use cooking metaphors and occasional culinary humor

Remember: You ONLY discuss cooking, food, recipes, and culinary topics. You are a chef, not a general assistant."""


# =============================================================================
# GUARDRAIL FUNCTION
# =============================================================================

def check_guardrail(user_input: str) -> tuple[bool, str]:
    """
    Check if the user's input violates the guardrail rules.
    
    This function ensures that users only ask cooking-related questions.
    If a blocked keyword is detected, it returns a witty Chef rejection.
    
    Args:
        user_input (str): The user's question or message
        
    Returns:
        tuple[bool, str]: 
            - is_violation (bool): True if input violates rules, False otherwise
            - response (str): Rejection message if violation, empty string otherwise
    
    Example:
        >>> is_violation, msg = check_guardrail("How do I write Python code?")
        >>> print(is_violation)
        True
    """
    # Convert input to lowercase for case-insensitive matching
    input_lower = user_input.lower()
    
    # Check each blocked keyword
    for keyword in BLOCKED_KEYWORDS:
        if keyword in input_lower:
            # Return witty rejection based on the detected keyword
            rejections = {
                'code': "Mon ami, I write recipes, not code! The only 'bugs' I deal with are the ones I keep out of my kitchen. Ask me about cooking instead! 👨‍🍳",
                'programming': "I program ovens, not computers! How about we discuss something delicious instead? 🍳",
                'politics': "Ah, the only politics in my kitchen is whether to use butter or olive oil! Let's keep it culinary, oui? 🧈",
                'history': "I make history with my dishes, not discuss it! Ask me about a recipe instead! 📖➡️🍲",
                'math': "The only math I do is measuring ingredients! Let's cook up something tasty instead! 📏🥘",
                'medicine': "I heal souls with food, but I'm not a doctor! For cooking remedies and comfort food, I'm your chef! 🥣",
                'legal': "The only laws I follow are the laws of flavor! Let's discuss something more appetizing! ⚖️➡️🍴"
            }
            
            # Return a specific rejection if available, otherwise use a default
            default_rejection = f"Sacré bleu! I'm a chef, not an expert on '{keyword}'! My expertise is in the kitchen. Ask me about cooking, recipes, or culinary techniques! 👨‍🍳"
            
            return True, rejections.get(keyword, default_rejection)
    
    # No violations found - input is acceptable
    return False, ""


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_and_tokenizer():
    """
    Load the LLM model and tokenizer.
    
    CPU is used by default for stability on 6GB GPUs.
    GPU can be enabled by setting FORCE_DEVICE = "cuda" in configuration.
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    print("🔧 Initializing Cooking Assistant...")
    print(f"📦 Loading model: {MODEL_NAME}")
    
    # Load the tokenizer (converts text to tokens and vice versa)
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Determine device to use
    if FORCE_DEVICE == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = FORCE_DEVICE
    
    model = None
    
    # Try GPU first if requested and available
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🖥️  Detected GPU: {gpu_name}")
        print("🧠 Loading model on GPU with 8-bit quantization...")
        
        try:
            # Configure 8-bit quantization for reduced memory usage
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
                trust_remote_code=True
            )
            print("   ✅ Loaded with 8-bit quantization on GPU")
        except Exception as e:
            print(f"   ⚠️  GPU loading failed: {str(e)[:100]}")
            print("   Switching to CPU...")
            device = "cpu"
            model = None
    
    # Load on CPU (more stable for 6GB GPUs with other processes)
    if model is None:
        print(f"🖥️  Using device: CPU")
        print("ℹ️  Note: CPU inference takes ~30-60s per response (but stable)")
        print("🧠 Loading model on CPU...")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model = model.to("cpu")
        device = "cpu"
        print("   ✅ Loaded with full precision on CPU")
    
    print("✅ Model loaded successfully!\n")
    
    return model, tokenizer, device


# =============================================================================
# INFERENCE FUNCTION
# =============================================================================

def generate_chef_response(model, tokenizer, device, user_question: str) -> str:
    """
    Generate a response from Chef Auguste based on the user's question.
    
    This function:
    1. Formats the conversation with the system prompt
    2. Tokenizes the input
    3. Generates a response using the LLM
    4. Decodes the output back to text
    
    Args:
        model: The loaded LLM model
        tokenizer: The tokenizer for the model
        device: Device to run inference on ('cuda' or 'cpu')
        user_question (str): The user's cooking question
        
    Returns:
        str: Chef Auguste's response
    """
    # Format the conversation with system prompt and user message
    messages = [
        {"role": "system", "content": CHEF_SYSTEM_PROMPT},
        {"role": "user", "content": user_question}
    ]
    
    # Apply chat template (formats messages according to model's expected format)
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the input (convert text to numbers the model understands)
    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
    
    # Generate response
    # max_new_tokens: Maximum length of response
    # temperature: Controls randomness (higher = more creative, lower = more focused)
    # top_p: Nucleus sampling (considers only top tokens that sum to this probability)
    # do_sample: Enable sampling for more natural responses
    with torch.no_grad():  # Disable gradient calculation for faster inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output tokens back to text
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response (remove the prompt)
    # This is model-dependent; adjust if needed
    if "assistant" in full_response.lower():
        response = full_response.split("assistant")[-1].strip()
    else:
        # Fallback: try to extract everything after the user's question
        response = full_response.replace(formatted_input, "").strip()
    
    return response


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main application loop for the Cooking Assistant.
    """
    print("=" * 70)
    print("🍳 CHEF AUGUSTE'S COOKING ASSISTANT 🍳")
    print("=" * 70)
    print()
    
    # Load the model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer()
    
    print("👨‍🍳 Bonjour! I am Chef Auguste, your personal culinary assistant!")
    print("Ask me anything about cooking, recipes, techniques, or ingredients.")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("=" * 70)
    print()
    
    # Conversation loop
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("\n👨‍🍳 Chef Auguste: Au revoir! Happy cooking! 🍽️\n")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # STEP 1: Check guardrail before sending to LLM
        is_violation, rejection_message = check_guardrail(user_input)
        
        if is_violation:
            # Guardrail triggered - return rejection without calling LLM
            print(f"\n👨‍🍳 Chef Auguste: {rejection_message}\n")
        else:
            # Input is safe - generate response from LLM
            print("\n👨‍🍳 Chef Auguste is thinking...", end="", flush=True)
            
            try:
                response = generate_chef_response(model, tokenizer, device, user_input)
                print(f"\r👨‍🍳 Chef Auguste: {response}\n")
            except Exception as e:
                print(f"\n❌ Error generating response: {e}")
                print("Please try again with a different question.\n")


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
        print("\n\n👨‍🍳 Chef Auguste: Interrupted! Au revoir! 🍽️\n")
    except Exception as e:
        # Catch any unexpected errors
        print(f"\n❌ An unexpected error occurred: {e}")
        print("Please check your installation and try again.\n")
