# Giktek Knowledge Base RAG Assistant 🚀

An enterprise-grade, privacy-first Retrieval-Augmented Generation (RAG) pipeline built to index dynamic internal company knowledge base documents (e.g., Outline) and answer employee queries directly within **Zoho Cliq**.

The architecture decouples vector retrieval and local LLM inference from synchronous webhook endpoints, using background worker tasks to ensure instant Zoho Cliq response times without timeout failures.

---

## 🏗 System Architecture

```
                                +-----------------------------+
                                |  Outline Knowledge Base     |
                                |  (Dynamic JS/React Pages)   |
                                +--------------+--------------+
                                               |
                                               | PlaywrightURLLoader
                                               v
                                +--------------+--------------+
                                |  ingest.py (Chunking & MD5) |
                                +--------------+--------------+
                                               |
                                               | nomic-embed-text
                                               v
                                +--------------+--------------+
                                |  Qdrant Vector Database     |
                                |  (Docker / Port 6333)       |
                                +--------------+--------------+
                                               ^
                                               | Similarity Search (k=5)
+-------------------+      HTTP POST           |
|  Zoho Cliq User   | ----------------> +-------+---------------------+
|  (Bot DM / Chat)  |                   |  FastAPI Webhook Server     |
+-------------------+ <---------------- |  (Async Background Worker)  |
                           Bot API POST +-------+---------------------+
                                               |
                                               | Context + Prompt
                                               v
                                +--------------+--------------+
                                |  Ollama (llama3.2 Local LLM)|
                                |  (Docker / Port 11434)      |
                                +-----------------------------+

```

---

## ✨ Features & Technical Highlights

* **Dynamic Web Scraping:** Uses `PlaywrightURLLoader` with `wait_until="networkidle"` to fully render React/Next.js dynamic single-page applications before chunking.
* **Deterministic & Idempotent Vector Indexing:** Employs content + source MD5 hashing to generate unique chunk IDs, enabling clean, duplicate-free syncs in Qdrant.
* **Asynchronous Webhook Engine:** FastAPI returns an instant HTTP `200 OK` to Zoho Cliq, executing vector search and local inference in background worker tasks to prevent network timeouts.
* **Term Equivalence & Prompt Engineering:** Configured with domain-specific rule mapping (e.g., equating *"test unit coverage"* with *"code coverage"*) and `temperature=0.0` for deterministic, factual outputs.
* **Persistent Model Pinning:** Utilizes `keep_alive="-1"` in Ollama to maintain embedding and generation models in memory/VRAM, eliminating cold-start latencies.

---

## 🛠 Project Structure

```
.
├── ingest.py           # Document scraper, text splitter, and Qdrant indexing pipeline
├── main.py             # FastAPI server, background worker, RAG chain & Zoho Cliq webhook
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

```

---

## 💻 Environment Setup (Local Infrastructure)

### Prerequisites

* **Python:** 3.10+
* **Docker & Docker Desktop:** Installed and running.
* **PowerShell / Terminal**

---

### Step 1: Start Docker Services

Run local Docker containers for **Qdrant** and **Ollama**:

```bash
# Start Qdrant Vector Database
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant

# Start Ollama Engine
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

```

Pull the required local embedding and LLM models via Ollama CLI inside the container:

```bash
docker exec -it ollama ollama pull nomic-embed-text
docker exec -it ollama ollama pull llama3.2

```

---

### Step 2: Install Python Dependencies

Create and activate a Python virtual environment:

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\activate

```

Install the dependencies and chromium binary:

```powershell
pip install -q playwright unstructured nest_asyncio langchain-community langchain-core langchain-text-splitters langchain-ollama langchain-qdrant fastapi uvicorn requests pydantic
python -m playwright install chromium

```

---

### Step 3: Run Document Ingestion (`ingest.py`)

Ensure your target Outline URLs are populated in `ingest.py`, then run:

```bash
python ingest.py

```

---

### Step 4: Launch the FastAPI Webhook Server

Start the API server on port `8000`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```

---

### Step 5: Expose Local Server with ngrok (For Public Access)

**ngrok** creates a secure public URL that tunnels to your local FastAPI server, allowing Zoho Cliq to access your webhook during development.

#### Install ngrok

Download and install ngrok from [ngrok.com](https://ngrok.com/download):

**Windows:**
```powershell
# Using Chocolatey
choco install ngrok

# Or download manually and add to PATH
# https://ngrok.com/download
```

**macOS:**
```bash
brew install ngrok/ngrok/ngrok
```

**Linux:**
```bash
# Download and extract
curl -s https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip | unzip -
sudo mv ngrok /usr/local/bin/
```

#### Start ngrok Tunnel

In a new terminal, expose your local FastAPI server to the public internet:

```bash
ngrok http 8000
```

Expected output:
```
ngrok                                                              (Ctrl+C to quit)

Session Status                online
Account                       your-account@example.com
Version                       3.x.x
Region                        us-central (US)
Latency                        0 ms
Web Interface                  http://127.0.0.1:4040
Forwarding                     https://abc123xyz.ngrok.io -> http://localhost:8000
```

**Copy the `Forwarding` URL** — this is your public webhook URL.

#### Keep ngrok Running

ngrok only forwards traffic while it's running. Open a new terminal and keep it active in the background:

```bash
ngrok http 8000
```

The tunnel will remain active as long as this terminal stays open. For persistent long-term deployment, consider using a cloud platform or ngrok Pro (which provides static URLs and higher limits).

---

### Step 6: Configure Zoho Cliq Integration

1. In **Zoho Cliq**, navigate to **Bots** -> Create a new Bot (e.g., `GiktekAssistant`).
2. Obtain your **Incoming Bot API Webhook URL** / `zapikey`.
3. Configure your Bot Handler in Deluge to forward incoming chat messages via `postUrl` (with `isSync = false`) to your FastAPI backend endpoint using the **ngrok public URL**:
   ```
   https://abc123xyz.ngrok.io/chat
   ```
   Replace `abc123xyz.ngrok.io` with your actual ngrok URL from Step 5.

#### Example Deluge Configuration

```javascript
// In Zoho Cliq Bot Handler
map<string, string> headers = new map<string, string>();
headers.put("Content-Type", "application/json");

map<string, string> payload = new map<string, string>();
payload.put("query", queryText);
payload.put("user_id", userEmail);
payload.put("user_email", userEmail);

// Use ngrok URL here
response = invokeurl(
    [
        URL: "https://abc123xyz.ngrok.io/chat",
        TYPE: POST,
        PARAMETERS: payload,
        HEADERS: headers,
        CONNECTION: "zoho_cliq_connection"
    ]
);
```

#### Webhook Verification

Test your webhook with curl:

```bash
# Test with ngrok URL
curl -X POST https://abc123xyz.ngrok.io/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is our deployment process?", "user_id": "user@company.com", "user_email": "user@company.com"}'
```

#### ngrok Free Tier Limitations

- **Session duration:** 2 hours (auto-restarts with new URL)
- **Bandwidth:** 1 GB/month
- **Connections:** 20/min
- **Dynamic URL:** URL changes on restart (paid plans have static URLs)

**For production, use ngrok Pro** (static URLs, higher limits, custom domains) or deploy to a cloud platform.

---

## ☁️ Cloud Architecture & Migration Guide

If you wish to scale this project beyond local execution, swap local services for managed cloud resources:

### 1. Vector Database: Managed Qdrant Cloud

* **Replace:** Local Docker Qdrant (`http://localhost:6333`)
* **With:** [Qdrant Cloud Managed Cluster](https://cloud.qdrant.io)
* **Code Update in `ingest.py` & `main.py`:**
```python
import os

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333",
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name="giktek_knowledge_base",
    vector_name="giktek-dense-vector"
)

```



### 2. Embeddings & LLM: Cloud API Providers

* **Replace:** Local Ollama (`nomic-embed-text` & `llama3.2`)
* **With:** OpenAI API, Cohere, or AWS Bedrock
* **Dependencies:** `pip install langchain-openai`
* **Code Update:**
```python
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY")
)

```



### 3. Webhook Backend Hosting

* **Deployment Platforms:** AWS ECS / Fargate, Render, DigitalOcean App Platform, or GCP Cloud Run.
* **Environment Variables:**
* `QDRANT_URL`
* `QDRANT_API_KEY`
* `OPENAI_API_KEY`
* `CLIQ_MESSAGE_WEBHOOK`



---

## ❓ Troubleshooting & Edge Cases

### Local Development Issues

* **Empty Scraped Content (`Content Length: 0`):** Ensure `PlaywrightURLLoader` includes `wait_until="networkidle"`. Single Page Applications (SPAs) require time for JavaScript rendering to complete.
* **Playwright Async Loop Error:** In `.py` scripts, wrap Playwright calls inside an `async def main()` coroutine executed with `asyncio.run(main())`.
* **Low Retrieval Matching:** Set `score_threshold` between `0.3` and `0.4` for `nomic-embed-text` embeddings, and ensure the LLM prompt explicitly handles technical domain synonyms.

### ngrok Issues

| Issue | Solution |
|-------|----------|
| **"Connection refused" from Zoho Cliq** | Ensure FastAPI is running on port 8000 and ngrok tunnel is active. Check `http://localhost:4040` (ngrok dashboard). |
| **ngrok URL expired/changed** | Free tier URLs expire after 2 hours. Save the URL somewhere or upgrade to ngrok Pro for persistent URLs. Restart ngrok if needed. |
| **"Failed to connect" in ngrok logs** | Make sure `uvicorn main:app --host 0.0.0.0 --port 8000` is running. Test locally first: `curl http://localhost:8000/health`. |
| **Zoho Cliq webhook keeps timing out** | Ensure async background processing is enabled in `main.py`. FastAPI should return HTTP 200 immediately, not wait for LLM response. |
| **ngrok session quota exceeded** | You've hit the free tier limit (1 GB/month or 20 connections/min). Upgrade to ngrok Pro or wait for quota reset. |
| **"connection refused" locally** | Make sure FastAPI is running with `uvicorn main:app --host 0.0.0.0 --port 8000 --reload` first. |
| **HTTP 403 from Zoho Cliq** | Verify the ngrok URL in your Zoho Cliq bot handler matches the active tunnel. Check `ngrok logs` for rejected requests. |

---

## 📜 License

Internal Enterprise Tooling for Giktek. All rights reserved.