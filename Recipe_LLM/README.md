# 🍳 Chef Auguste's Cooking Assistant

A local LLM-based cooking assistant built for educational workshops. This project demonstrates how to run a lightweight Large Language Model (4B parameter range) entirely on your local machine, with custom guardrails and persona implementation.

## 📋 Overview

This workshop project teaches you how to:
- Load and run a local LLM using the `transformers` library
- Implement 4-bit quantization to reduce memory usage
- Create a persona using system prompts
- Add guardrails to filter inappropriate inputs
- Handle GPU/CPU fallback gracefully

## 🎯 Features

- **Local LLM**: Runs entirely on your machine (no API keys needed!)
- **Chef Persona**: Helpful cooking assistant with personality
- **Smart Guardrails**: Filters non-cooking topics before LLM inference
- **Memory Efficient**: Uses 4-bit quantization (~2-4GB RAM)
- **Well-Documented**: Code comments explain every step for learning

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv cooking_assistant_env
source cooking_assistant_env/bin/activate  # Linux/Mac
# OR: cooking_assistant_env\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

See [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) for detailed installation steps.

### 2. Run the Assistant

```bash
python cooking_assistant.py
```

The first run will download the model (~2-4GB).

## 💡 Example Interaction

```
You: How do I make perfect scrambled eggs?

👨‍🍳 Chef Auguste: Ah, the art of scrambled eggs! Here's my technique:
1. Crack 3 eggs into a bowl, add a splash of milk and salt
2. Whisk until just combined - don't overbeat!
3. Heat butter in a non-stick pan over medium-low heat
4. Pour in eggs and let sit for 20 seconds
5. Gently fold with a spatula - patience is key!
6. Remove from heat while still slightly wet (they'll continue cooking)

The secret? Low heat and patience. Bon appétit! 🍳

You: Can you help me write Python code?

👨‍🍳 Chef Auguste: Mon ami, I write recipes, not code! The only 'bugs' I 
deal with are the ones I keep out of my kitchen. Ask me about cooking instead! 👨‍🍳
```

## 🛡️ Guardrail System

The guardrail function checks user input **before** calling the LLM, filtering out non-cooking topics:

- Programming/code questions
- Politics, history, legal topics  
- Medical advice
- Mathematics
- And more...

This demonstrates a simple but effective approach to keeping the assistant on-topic.

## 📚 Educational Notes

### System Prompts
The `CHEF_SYSTEM_PROMPT` defines the assistant's personality and behavior. Try modifying it to create different personas!

### 4-bit Quantization
Reduces model size by ~75% with minimal quality loss:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
```

### Model Options
The script supports multiple models:
- `Qwen/Qwen2.5-3B-Instruct` (recommended)
- `microsoft/Phi-3.5-mini-instruct`

Change by modifying the `MODEL_NAME` variable.

## 🔧 Customization Ideas

1. **Add More Guardrails**: Extend `BLOCKED_KEYWORDS` with your own filters
2. **Change Persona**: Modify `CHEF_SYSTEM_PROMPT` to create a different character
3. **Adjust Generation**: Experiment with `temperature` and `top_p` parameters
4. **Add Memory**: Implement conversation history to remember context
5. **Multi-turn Conversations**: Store previous messages in the `messages` list

## 📊 System Requirements

- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **Python**: 3.8 or higher
- **Disk Space**: ~10GB (for model and dependencies)

## 🐛 Troubleshooting

**Out of Memory?**
- Close other applications
- Use CPU mode (slower but works)
- Try the smaller Qwen model

**Slow responses?**
- GPU highly recommended
- On CPU, expect 30-60 seconds per response

**Installation issues?**
- Check Python version: `python --version`
- Update pip: `pip install --upgrade pip`
- See [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)

## 📖 Learning Resources

- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization)
- [Prompt Engineering Tips](https://www.promptingguide.ai/)

## 📝 License

Educational use - feel free to modify and share!

## 🙏 Acknowledgments

- Built with Hugging Face Transformers
- Uses Qwen2.5 or Phi-3.5 models
- Created for educational workshops

---

**Happy Cooking (with AI)! 👨‍🍳🤖**
