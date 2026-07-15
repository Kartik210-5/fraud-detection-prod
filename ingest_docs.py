# ingest_docs.py
import os
from pathlib import Path
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from openai import OpenAI
from supabase import create_client

load_dotenv()

# Initialize Clients
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = OpenAI()
firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))


def get_embedding(text: str) -> list:
    """Generates a 1536-dim embedding."""
    cleaned = text.replace("\n", " ")
    response = openai_client.embeddings.create(
        input=[cleaned], model="text-embedding-3-small"
    )
    return response.data[0].embedding


def save_chunk(source_file: str, url: str | None, content: str):
    """Generates embedding and saves chunk to PostgreSQL."""
    embedding = get_embedding(content)
    data = {
        "source_file": source_file,
        "source_url": url,
        "content": content,
        "embedding": embedding,
    }
    supabase.table("doc_chunks").insert(data).execute()
    print(f"✅ Indexed chunk from: {source_file}")


# 1. Ingest Local System Docs (e.g., README.md, reflections)
def ingest_local_files():
    targets = ["README.md", "model-card.md", "reflections.md"]
    for filename in targets:
        path = Path(filename)
        if path.exists():
            content = path.read_text()
            # Simple chunking by paragraph/section
            chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 100]
            for chunk in chunks:
                save_chunk(filename, None, chunk)


# 2. Scrape External Pages using Firecrawl
def ingest_external_docs():
    urls = [
        "https://docs.evidentlyai.com/reference/data-drift",
        "https://fastapi.tiangolo.com/tutorial/security/",
    ]
    for url in urls:
        print(f"🕷️ Scraping {url} with Firecrawl...")
        # Firecrawl returns clean, LLM-ready markdown format
        result = firecrawl_app.scrape_url(url, params={"formats": ["markdown"]})
        markdown_content = result.get("markdown", "")

        # Split markdown into clean structural chunks
        chunks = [
            c.strip() for c in markdown_content.split("##") if len(c.strip()) > 100
        ]
        for idx, chunk in enumerate(chunks):
            # Prepend the ## back for context
            formatted_chunk = f"## {chunk}" if idx > 0 else chunk
            save_chunk(url, url, formatted_chunk)


if __name__ == "__main__":
    ingest_local_files()
    ingest_external_docs()