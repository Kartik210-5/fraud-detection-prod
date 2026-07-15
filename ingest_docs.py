# ingest_docs.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
# Bypass TensorFlow import to prevent Segmentation Faults/Permission Errors on macOS
sys.modules['tensorflow'] = None

# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
import torch
from supabase import create_client

# pyrefly: ignore [missing-import]
from firecrawl import Firecrawl

load_dotenv()

# =====================================================================
# SYSTEM DEVICE MATCHING
# =====================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍏 Apple Silicon detected. Using GPU for local embedding generation!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🔥 NVIDIA GPU detected. Using CUDA!")
else:
    device = torch.device("cpu")
    print("💻 Running on CPU.")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =====================================================================
# INITIALIZE CLIENTS & LOCAL EMBEDDING MODEL
# =====================================================================
print("⏳ Loading local embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

if not supabase_url or not supabase_key:
    print("❌ Error: Missing SUPABASE_URL or SUPABASE_KEY in environment.")
    sys.exit(1)

supabase = create_client(supabase_url, supabase_key)

# Initialize modern Firecrawl client
firecrawl_app = Firecrawl(api_key=firecrawl_api_key) if firecrawl_api_key else None


# =====================================================================
# INGESTION PIPELINE FUNCTIONS
# =====================================================================
def get_local_embedding(text: str) -> list:
    """Generates a local 384-dim embedding using sentence-transformers."""
    cleaned = text.replace("\n", " ")
    vector = embedding_model.encode(cleaned).tolist()
    return vector


def save_chunk(source_file: str, url: str | None, content: str):
    """Generates a local embedding and saves the chunk to Supabase (pgvector)."""
    try:
        embedding = get_local_embedding(content)
        data = {
            "source_file": source_file,
            "source_url": url,
            "content": content,
            "embedding": embedding,
        }
        supabase.table("doc_chunks").insert(data).execute()
        print(f"✅ Indexed chunk from: {source_file}")
    except Exception as e:
        print(f"❌ Failed to index chunk from {source_file}: {e}")


# 1. Ingest Local System Docs (ONLY README.md & model-card.md - no reflections!)
def ingest_local_files():
    targets = ["README.md", "model-card.md"]  # 👈 Removed reflections.md from target loop
    for filename in targets:
        path = Path(filename)
        if path.exists():
            print(f"📂 Processing local document: {filename}...")
            content = path.read_text()
            chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 100]
            for chunk in chunks:
                save_chunk(filename, None, chunk)
        else:
            print(f"⚠️ Warning: Local target file '{filename}' was not found.")


# 2. Scrape External Pages using Firecrawl
def ingest_external_docs():
    if not firecrawl_app:
        print("⏭️ Skipping Firecrawl scraping: FIRECRAWL_API_KEY not found in environment.")
        return

    urls = [
        "https://docs.evidentlyai.com/reference/data-drift",
        "https://fastapi.tiangolo.com/tutorial/security/",
    ]
    for url in urls:
        print(f"🕷️ Scraping {url} with Firecrawl...")
        try:
            # Modern Firecrawl API syntax:
            result = firecrawl_app.scrape(url, formats=["markdown"])
            
            # Extract the raw markdown text attribute from the Document object directly
            markdown_content = result.markdown if hasattr(result, "markdown") else ""

            if not markdown_content:
                print(f"⚠️ No markdown content recovered from {url}")
                continue

            # Split markdown into clean, header-based blocks
            chunks = [
                c.strip() for c in markdown_content.split("##") if len(c.strip()) > 100
            ]
            for idx, chunk in enumerate(chunks):
                formatted_chunk = f"## {chunk}" if idx > 0 else chunk
                save_chunk(url, url, formatted_chunk)
        except Exception as e:
            print(f"❌ Failed to scrape {url} via Firecrawl: {e}")


if __name__ == "__main__":
    print("🚀 Initiating local-first vector ingestion...")
    ingest_local_files()
    ingest_external_docs()
    print("✨ Vector ingestion loop completed successfully!")