# HR RAG Assistant - Setup Instructions

Complete setup guide for the HR RAG demonstration.

## Prerequisites
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- Optional: NVIDIA GPU with 4GB+ VRAM

## Step 1: Create Virtual Environment

```bash
# Create environment
python3 -m venv rag_env

# Activate environment
# On Linux/Mac:
source rag_env/bin/activate

# On Windows:
# rag_env\Scripts\activate
```

## Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose based on your system)
# For CUDA 11.8 (GPU):
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch==2.5.0

# Install all other dependencies
pip install -r requirements.txt
```

## Step 3: Verify Installation

```bash
# Test imports
python -c "import langchain; import chromadb; import sentence_transformers; print('All imports successful!')"
```

## Step 4: Run the Assistant

```bash
python hr_rag_assistant.py
```

## First Run Behavior

The first time you run the script, it will:

1. **Download Models** (~2-3 minutes):
   - Sentence transformer for embeddings (~400MB)
   - Qwen2-1.5B-Instruct LLM (~3GB)

2. **Create Vector Database** (~10 seconds):
   - Load employee_records.csv
   - Generate embeddings for all records
   - Build ChromaDB index
   - Save to ./chroma_db/

3. **Run Example Queries** (~30-60 seconds each):
   - Promotion candidates
   - Performance improvement needs

4. **Enter Interactive Mode**:
   - Ask custom HR questions
   - Type 'quit' to exit

## Configuration Options

### Use CPU Instead of GPU

Edit `hr_rag_assistant.py`:
```python
DEVICE = "cpu"  # Force CPU mode
```

### Use Different Models

**Smaller Embedding Model** (faster, less accurate):
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Current
# OR
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # Smaller
```

**Different LLM** (if you have more GPU memory):
```python
LLM_MODEL = "Qwen/Qwen2-1.5B-Instruct"  # Current (1.5B)
# OR
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # Larger (3B)
```

### Adjust Retrieval Count

Edit the retriever setup:
```python
retriever = vectordb.as_retriever(
    search_kwargs={"k": 10}  # Retrieve top 10 instead of 5
)
```

## Troubleshooting

### Issue: ChromaDB Permission Error

**Solution:**
```bash
rm -rf ./chroma_db
# Database will be recreated on next run
```

### Issue: Out of Memory (GPU)

**Solution 1:** Use CPU mode
```python
DEVICE = "cpu"
```

**Solution 2:** Close other GPU applications
```bash
nvidia-smi  # Check GPU memory usage
```

### Issue: Slow Embeddings Creation

**Solution:** This is normal on first run. Subsequent runs reuse the database.

If too slow, use a smaller embedding model:
```python
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
```

### Issue: "No module named 'langchain'"

**Solution:**
```bash
pip install langchain langchain-community
```

### Issue: ChromaDB SQLite Error

**Solution:**
```bash
pip install --upgrade chromadb pysqlite3-binary
```

## Performance Tips

### For Faster Responses
- Use GPU mode (if available)
- Reduce `max_new_tokens` to 256
- Lower retrieval count to k=3

### For Better Quality
- Increase `max_new_tokens` to 1024
- Use larger embedding model
- Increase retrieval count to k=10
- Lower LLM temperature to 0.1 (more focused)

## Adding Your Own Data

### Modify CSV Structure

1. Edit `employee_records.csv`
2. Keep the same columns or update the document creation function
3. Delete `./chroma_db/` folder
4. Run the script again

### Custom Questions

In interactive mode, try:
```
- "Who are the top 3 performers?"
- "Which employees have attendance issues?"
- "Who shows leadership potential?"
- "Which employees received negative feedback?"
```

## Workshop Notes

- **First run**: Takes 3-5 minutes (model downloads)
- **Subsequent runs**: Takes 30-60 seconds (models cached)
- **Query time**: 10-30 seconds per question (depending on CPU/GPU)
- **Database**: Persists between runs, no need to rebuild

## Clean Start

To completely reset and start fresh:

```bash
# Remove vector database
rm -rf ./chroma_db

# Clear model cache (optional)
rm -rf ~/.cache/huggingface

# Run again
python hr_rag_assistant.py
```

## System Resource Usage

**CPU Mode:**
- RAM: ~4-6GB
- CPU: High usage during inference
- Disk: ~5GB for models

**GPU Mode:**
- VRAM: ~3-4GB (with 8-bit quantization)
- RAM: ~2-3GB
- Disk: ~5GB for models

## Next Steps

After successful setup:
1. Try the example queries
2. Experiment with custom questions in interactive mode
3. Modify the CSV with your own data
4. Adjust the prompt template for different use cases
5. Explore different embedding and LLM models

---

**Happy Learning! 🎓**
