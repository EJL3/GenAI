# 🎯 HR RAG Assistant - Employee Promotion Analysis

An advanced Retrieval-Augmented Generation (RAG) system for HR decision-making using LangChain, ChromaDB, and local LLMs.

## 📋 Overview

This workshop project demonstrates:
- **RAG Architecture**: Combining retrieval with generation for informed responses
- **Vector Databases**: Using ChromaDB for semantic search over employee data
- **Local Embeddings**: HuggingFace sentence-transformers for text embeddings
- **LangChain Integration**: Building production-ready RAG chains
- **Decision Logic**: Transparent reasoning for HR recommendations

## 🎯 Features

✅ **Load Employee Data** - Parse CSV with attendance, performance, and feedback  
✅ **Create Embeddings** - Local HuggingFace models (no API needed)  
✅ **Vector Search** - ChromaDB for semantic similarity retrieval  
✅ **RAG Chain** - LangChain RetrievalQA for intelligent responses  
✅ **Decision Reasoning** - Explainable AI for HR recommendations  
✅ **Interactive Mode** - Ask custom HR questions in real-time

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv rag_env
source rag_env/bin/activate  # Linux/Mac
# OR: rag_env\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### 2. Run the Assistant

```bash
python hr_rag_assistant.py
```

The script will:
1. Load 10 employee records from CSV
2. Create embeddings and build vector database
3. Initialize local LLM (Qwen2-1.5B)
4. Run example queries
5. Enter interactive mode for custom questions

## 📊 How It Works

### 1. Data Loading
```python
# Load employee CSV
df = pd.read_csv("employee_records.csv")
```

### 2. Document Creation
Each employee record becomes a structured document:
```
Employee ID: E001
Attendance Rate: 98.0%
Performance Score: 9/10
Manager Feedback: Sarah consistently exceeds expectations...
```

### 3. Embeddings & Vector DB
```python
# Create embeddings using local model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build ChromaDB vector store
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)
```

### 4. RAG Chain
```python
# Retrieve top 5 relevant employees
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Chain retrieval + generation
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)
```

### 5. Query & Response
```python
response = qa_chain.invoke({
    "query": "Which employees deserve a promotion?"
})
```

## 💡 Example Queries

### Promotion Candidates
```
Question: Which employees deserve a promotion based on their feedback and performance?

Answer:
1. Emily (E005) - Outstanding performer, 99% attendance, 10/10 score
2. Sarah (E001) - Exceeds expectations, 98% attendance, 9/10 score
3. Amanda (E009) - Leadership potential, 96% attendance, 9/10 score

Decision Logic:
- All three have attendance >95% (excellent reliability)
- Performance scores 9-10/10 (top tier)
- Manager feedback includes: "exceeds expectations", "leadership", "innovation"
```

### Performance Improvement
```
Question: Which employees need performance improvement plans?

Answer:
1. Robert (E008) - 65% attendance, 4/10 score, inconsistent quality
2. David (E004) - 72% attendance, 5/10 score, reliability concerns

Decision Logic:
- Attendance <75% (below acceptable threshold)
- Performance scores <5/10 (needs improvement)
- Manager feedback mentions: "struggles", "inconsistent", "requires supervision"
```

## 🔧 Configuration

### Models Used

**Embedding Model:**
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Fast, efficient, 384-dimensional embeddings
```

**LLM Model:**
```python
LLM_MODEL = "Qwen/Qwen2-1.5B-Instruct"
# Small (1.5B), fast, runs on 6GB GPU or CPU
```

### Customization

**Change Retrieval Count:**
```python
retriever = vectordb.as_retriever(
    search_kwargs={"k": 10}  # Retrieve top 10 instead of 5
)
```

**Adjust LLM Temperature:**
```python
pipe = pipeline(
    "text-generation",
    temperature=0.7,  # Higher = more creative, lower = more focused
    max_new_tokens=1024  # Longer responses
)
```

## 📁 Project Structure

```
RAG_LLM/
├── employee_records.csv       # Employee data (10 records)
├── hr_rag_assistant.py        # Main RAG script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── SETUP_INSTRUCTIONS.md      # Detailed setup guide
└── chroma_db/                 # Vector database (auto-created)
```

## 🎓 Educational Features

### RAG Architecture Explained

1. **Retrieval**: Find most relevant employee records using semantic similarity
2. **Augmentation**: Add retrieved context to the prompt
3. **Generation**: LLM generates informed response with reasoning

### Why RAG?

- ✅ **Accurate**: Responses based on actual data, not hallucinations
- ✅ **Transparent**: Can see which documents were used
- ✅ **Updatable**: Add new employees without retraining model
- ✅ **Explainable**: Shows decision logic and reasoning

## 📊 System Requirements

- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **Python**: 3.8 or higher
- **Disk Space**: ~5GB (for models and database)

## 🐛 Troubleshooting

**ChromaDB Permission Error?**
```bash
rm -rf ./chroma_db
# Database will be recreated on next run
```

**Out of Memory?**
```python
# Use CPU mode
DEVICE = "cpu"
```

**Slow Responses?**
- Reduce `max_new_tokens` to 256
- Use smaller embedding model
- Reduce retrieval count (`k=3`)

## 📖 Learning Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [ChromaDB Guide](https://docs.trychroma.com/)
- [RAG Explained](https://www.promptingguide.ai/techniques/rag)
- [Sentence Transformers](https://www.sbert.net/)

## 🎯 Workshop Use Cases

1. **HR Decision Support**: Promotion, performance reviews
2. **Customer Support**: Query product knowledge base
3. **Legal Research**: Search case law and precedents
4. **Medical Records**: Retrieve patient history
5. **Technical Documentation**: Find relevant code examples

## 📝 License

Educational use - free to modify and share!

---

**Built for AI Workshops - Demonstrating Production RAG Patterns! 🚀**
