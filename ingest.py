import os
import asyncio
import hashlib
from langchain_core.documents import Document
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

# Helper function for generating stable chunk IDs
def generate_chunk_id(doc: Document) -> str:
    """Generates a stable MD5 hash based on content and source metadata."""
    source = doc.metadata.get("source", "unknown")
    content = doc.page_content
    raw_string = f"{source}::{content}"
    return hashlib.md5(raw_string.encode("utf-8")).hexdigest()

async def main():
    documents = []

    urls = [
        "http://188.166.42.199:3000/share/ci5d7hsc10/p/objective-and-key-results-SKZlQ8ARTs",
        "http://188.166.42.199:3000/share/35v43o3gzs/p/developer-tools-myxZNg5A96"
    ]

    # 1. Load documents using Playwright
    web_loader = PlaywrightURLLoader(
        urls=urls,
        remove_selectors=["nav", "footer", "header"],
        wait_until="networkidle"  # Ensures React/JS dynamic content renders before reading text
    )

    # Correctly await the async coroutine
    web_docs = await web_loader.aload()
    documents.extend(web_docs)

    # 2. Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 3. Initialize Ollama Embeddings with persistent memory pinning
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
        keep_alive="-1"  # Keeps model loaded in VRAM to prevent cold-start latency
    )

    # 4. Generate deterministic chunk IDs
    chunk_ids = [generate_chunk_id(chunk) for chunk in chunks]

    # 5. Idempotent Sync to Qdrant Vector Store
    vector_db = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url="http://localhost:6333",
        collection_name="gitktek_knowledge_base",
        vector_name="giktek-dense-vector",
        ids=chunk_ids  # Overwrites matching chunk IDs, inserts new ones
    )

    print(f"Indexed {len(chunks)} chunks using Ollama embeddings into Qdrant.")

if __name__ == "__main__":
    asyncio.run(main())