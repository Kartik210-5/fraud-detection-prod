# local_rag.py
import os
import sys

# Bypass TensorFlow import to prevent Segmentation Faults/Permission Errors on macOS
sys.modules['tensorflow'] = None

import torch
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
from supabase import create_client
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()

# =====================================================================
# 1. ROBUST DEVICE DETECTION (macOS MPS / CUDA / CPU)
# =====================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(" Apple Silicon detected. Using GPU via MPS acceleration!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🔥 NVIDIA GPU detected. Using CUDA!")
else:
    device = torch.device("cpu")
    print("💻 No GPU acceleration available. Running on CPU.")

# Fix OpenMP conflict crashes on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =====================================================================
# 2. INITIALIZE LOCAL MODELS
# =====================================================================
print("⏳ Loading local embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

print("⏳ Loading local LLM (SmolLM2-1.7B-Instruct)...")
model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# CRITICAL FIX: Bypass device_map="auto" to prevent Segmentation Faults on Mac.
# We load the weights cleanly to CPU memory first, then transfer directly.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32  # Standard float32 is safest for MPS/CPU fallbacks
)
model = model.to(device)

# Build pipeline using explicitly mapped model and device
generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    device=device
)

# Initialize Supabase
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


# =====================================================================
# 3. LOCAL RAG PIPELINE
# =====================================================================
def ask_local_rag(question: str):
    # Step A: Generate local 384-dimensional embedding
    query_vector = embedding_model.encode(question).tolist()

    # Step B: Query Supabase pgvector RPC
    try:
        matched_res = (
            supabase.rpc(
                "match_chunks",
                {
                    "query_embedding": query_vector,
                    "match_threshold": 0.1,
                    "match_count": 3,
                },
            )
            .execute()
        )
        contexts = matched_res.data
    except Exception as e:
        print(f"❌ Supabase query failed: {e}")
        return

    if not contexts:
        print("I couldn't find any relevant system documents in the database.")
        return

    # Step C: Format context
    context_text = ""
    for idx, c in enumerate(contexts, 1):
        context_text += f"\n[Source {idx}]: {c['content']}\n"

    # Step D: Draft local model prompt using SmolLM2 Chat Template
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the following context to answer the user question. Keep your answer brief and factual.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {question}",
        },
    ]

    # Convert to Chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Step E: Generate Local Text
    print("\n🔮 Thinking (running local inference)...")
    outputs = generator(
        prompt, 
        max_new_tokens=250, 
        temperature=0.3, 
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract clean answer (removing prompt prefix)
    generated_text = outputs[0]["generated_text"]
    response = generated_text[len(prompt) :].strip()

    print("\n📝 --- ANSWER ---")
    print(response)
    print("\n📚 --- SOURCES ---")
    for idx, c in enumerate(contexts, 1):
        source = c["source_file"] if not c["source_url"] else c["source_url"]
        print(f"[{idx}] {source}")


if __name__ == "__main__":
    user_query = input("\n💬 Ask your system a question: ")
    ask_local_rag(user_query)