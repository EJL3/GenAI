import pandas as pd
import os
import warnings
from typing import List, Dict

# --- MODERN IMPORTS (Hardware/Models) ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
# These are now hosted in the classic package
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_core.documents import Document 

# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Suppress the endless warnings these libraries generate
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_FILE = "employee_records.csv"
CHROMA_DB_DIR = "./chroma_db_2026"  # Changed dir to avoid conflicts with old DBs
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2-1.5B-Instruct" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# DATA PROCESSING (Unchanged because it actually worked)
# =============================================================================

def load_and_process_data() -> List[Document]:
    """Loads CSV and converts to documents in one go."""
    if not os.path.exists(CSV_FILE):
        # Create dummy data if file is missing so the script doesn't crash immediately
        print("⚠️ CSV not found. Creating dummy data for demonstration...")
        data = {
            'Employee_ID': ['E001', 'E002'], 
            'Attendance_Rate': [0.95, 0.82],
            'Performance_Score_1to10': [9, 6],
            'Manager_Feedback': ['Exceptional leadership shown.', 'Struggles with deadlines.']
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(CSV_FILE)

    print(f"📂 Loaded {len(df)} records.")
    
    documents = []
    for _, row in df.iterrows():
        content = (
            f"Employee ID: {row['Employee_ID']}\n"
            f"Attendance: {row['Attendance_Rate']*100:.1f}%\n"
            f"Performance: {row['Performance_Score_1to10']}/10\n"
            f"Feedback: {row['Manager_Feedback']}"
        )
        # Metadata is crucial for filtering later
        meta = {"id": row['Employee_ID'], "score": int(row['Performance_Score_1to10'])}
        documents.append(Document(page_content=content, metadata=meta))
    
    return documents


# =============================================================================
# MODERN COMPONENT SETUP
# =============================================================================

def setup_vector_db(documents: List[Document]):
    """
    Uses langchain_chroma (the new standard) instead of community.
    """
    print(f"📦 Loading Embeddings: {EMBEDDING_MODEL}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': DEVICE}
    )

    print("🧮 Indexing data into ChromaDB...")
    # .from_documents is efficient for batch processing
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_DIR
    )
    return vectorstore


def setup_llm():
    """
    Sets up the local LLM using the modern HuggingFacePipeline wrapper.
    """
    print(f"🧠 Loading LLM: {LLM_MODEL} on {DEVICE.upper()}")
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    
    model_kwargs = {"trust_remote_code": True}
    if DEVICE == "cuda":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True, 
            llm_int8_threshold=6.0
        )
    
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, **model_kwargs)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,  # Kept low for factual HR data
        top_p=0.9,
        repetition_penalty=1.15  # Prevent the model from stuttering
    )
    
    return HuggingFacePipeline(pipeline=pipe)


# =============================================================================
# THE MODERN RAG CHAIN (LCEL)
# =============================================================================

def create_classic_rag_chain(vectorstore, llm):
    """
    Creates a RAG chain using the 'langchain-classic' library 
    (RetrievalQA) which you successfully installed.
    """
    # Import locally to ensure we use the classic versions
    from langchain_classic.chains import RetrievalQA
    from langchain_classic.prompts import PromptTemplate

    # 1. The Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. The Prompt (Classic uses standard PromptTemplate, not ChatPromptTemplate)
    prompt_template = """You are a critical HR assistant. 
Use the context below to answer the question. If you don't know, say so.

Context:
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 3. The Chain (RetrievalQA is the robust classic standard)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain


# =============================================================================
# EXECUTION
# =============================================================================

def main():
    print("--- HR RAG SYSTEM INITIALIZING ---")
    
    # Setup Data & Models
    docs = load_and_process_data()
    vectorstore = setup_vector_db(docs)
    llm = setup_llm()
    
    # Create the Chain
    rag_chain = create_classic_rag_chain(vectorstore, llm)
    
    print("\n✅ System Ready. Interactive Mode.\n")
    
    while True:
        query = input("Query (or 'q' to quit): ").strip()
        if query.lower() == 'q': break
        
        print("\n🤔 Thinking...")
        
        # Invoke the chain
        response = rag_chain.invoke({"query": query})
        
        print("\n" + "="*40)
        print("📝 ANSWER:")
        print(response["result"])
        print("\nREFERENCE DOCUMENTS:")
        for i, doc in enumerate(response["source_documents"]):
             print(f"[{i+1}] ID: {doc.metadata.get('id', 'N/A')} (Score: {doc.metadata.get('score', 'N/A')})")
        print("="*40 + "\n")

if __name__ == "__main__":
    main()