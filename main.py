import time
import requests
from typing import Optional
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

# LangChain & Qdrant Integration Imports
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
CLIQ_MESSAGE_WEBHOOK = "https://cliq.zoho.com/api/v2/bots/giktekassistant/message?zapikey=1001.643dcdd20d4a53a42f3c1fa1cedc994b.46bb1298a4865cf9e92a75c618a9d6cd"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "gitktek_knowledge_base"
VECTOR_NAME = "giktek-dense-vector"

app = FastAPI(title="Local Zoho Cliq RAG Backend")

# ==========================================
# PYDANTIC MODEL SCHEMAS
# ==========================================
class QueryPayload(BaseModel):
    query: str
    user_id: Optional[str] = None
    user_email: Optional[str] = None

# ==========================================
# RAG COMPONENTS INITIALIZATION
# ==========================================
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
    keep_alive="-1"
)

# Explicit float 0.0 forces strict deterministic responses
llm = ChatOllama(
    model="llama3.2",
    temperature=0.0,
    base_url="http://localhost:11434"
)

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
    vector_name=VECTOR_NAME,
)

retriever = vector_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.35
    }
)

template = """You are an internal AI assistant for Giktek.
Answer the user's question using ONLY the provided context below.

Rules:
1. Treat closely related technical terms as equivalent (e.g., 'unit test coverage' or 'test unit coverage' refers to 'code coverage').
2. If the context contains measurements or key results related to the query, state the exact percentage or target clearly.
3. If the context does not contain relevant information, state that you do not have this record in the company knowledge base.

Context:
{context}

Question: {question}

Answer:"""

prompt_template = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

# Direct processing chain (takes raw text context & question)
llm_chain = prompt_template | llm | output_parser

# ==========================================
# CLIQ DELIVERY HELPER
# ==========================================
def send_to_cliq_with_retry(payload: dict, max_attempts: int = 3):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                CLIQ_MESSAGE_WEBHOOK,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 200:
                return response
            last_error = f"status {response.status_code}: {response.text}"
        except requests.RequestException as e:
            last_error = str(e)

        print(f"[Cliq POST attempt {attempt} failed]: {last_error}, retrying...")
        time.sleep(attempt * 2)

    raise RuntimeError(f"Failed to deliver message to Cliq after {max_attempts} attempts: {last_error}")

# ==========================================
# BACKGROUND WORKER & SINGLE RETRIEVAL FLOW
# ==========================================
def process_rag_and_notify_cliq(query: str, user_id: Optional[str], user_email: Optional[str]):
    start_time = time.time()
    print(f"\n[Async Job Started] Query: '{query}'")

    recipient = user_email or user_id
    if not recipient:
        print("[Async Execution Error]: No user_id or user_email provided, cannot deliver response\n")
        return

    try:
        rag_start = time.time()
        
        # Single retrieval step to avoid redundant Qdrant vector calls
        retrieved_docs = retriever.invoke(query)
        
        if not retrieved_docs:
            answer = "I don't have this record in the company knowledge base."
            sources = []
        else:
            # Format context string
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Generate answer
            answer = llm_chain.invoke({"context": context_text, "question": query})
            
            # Extract distinct source URLs
            sources = list(set([doc.metadata.get("source", "Internal Knowledge Base") for doc in retrieved_docs]))

        rag_duration = time.time() - rag_start

        # Format output
        final_text = answer
        if sources and "don't have this record" not in answer:
            source_list_str = "\n".join([f"- {src}" for src in sources])
            final_text += f"\n\n*Sources Consulted:*\n{source_list_str}"

        webhook_payload = {
            "text": final_text,
            "userids": recipient
        }

        # Dispatch back to Zoho Cliq
        post_start = time.time()
        response = send_to_cliq_with_retry(webhook_payload)
        post_duration = time.time() - post_start
        total_duration = time.time() - start_time

        print(f"[Async Job Complete]")
        print(f" ├─ RAG Execution Time : {rag_duration:.2f} seconds")
        print(f" ├─ Cliq POST Time     : {post_duration:.2f} seconds")
        print(f" ├─ Total Time Elapsed : {total_duration:.2f} seconds")
        print(f" ├─ RAG Response       : {final_text}")
        print(f" └─ Cliq API Response  : Status {response.status_code} | Body: {response.text}\n")

    except Exception as e:
        print(f"[Async Execution Error]: {e}\n")

# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {"status": "Online", "engine": "Ollama + Qdrant Local RAG"}

@app.post("/chat")
async def chat_endpoint(payload: QueryPayload, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        process_rag_and_notify_cliq,
        payload.query,
        payload.user_id,
        payload.user_email
    )
    return {"status": "queued"}

# from fastapi import FastAPI
# from pydantic import BaseModel
# from langchain_qdrant import QdrantVectorStore
# from langchain_ollama import OllamaEmbeddings, ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# app = FastAPI()

# # 1. Initialize Ollama Embeddings (Must match the model used in ingest.py)
# embeddings = OllamaEmbeddings(
#     model="nomic-embed-text",
#     base_url="http://localhost:11434"
# )

# # 2. Connect to existing Qdrant collection
# vector_db = QdrantVectorStore.from_existing_collection(
#     embedding=embeddings,
#     url="http://localhost:6333",
#     collection_name="gitktek_knowledge_base",
#     vector_name="giktek-dense-vector"
# )

# retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# # 3. Initialize Ollama Chat LLM
# llm = ChatOllama(
#     model="llama3.2",  # or "mistral", "phi3"
#     base_url="http://localhost:11434",
#     temperature=0
# )

# # 4. Prompt Template
# prompt = ChatPromptTemplate.from_template("""
# You are a helpful company assistant. Answer the employee's question using ONLY the context provided below.
# If you don't know the answer, state that you don't have this record in the company knowledge base.

# Context:
# {context}

# Question: {input}
# """)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "input": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# class QueryPayload(BaseModel):
#     query: str
#     user_id: str
    
# @app.post("/chat")
# async def chat_endpoint(payload: QueryPayload):
#     answer = rag_chain.invoke(payload.query)
    
#     retrieved_docs = retriever.invoke(payload.query)
#     sources = list(set([doc.metadata.get("source", "Internal Doc") for doc in retrieved_docs]))
    
#     # Return a clean JSON object without custom card constructs
#     return {
#         "text": answer,
#         "sources": sources
#     }

# @app.post("/chat")
# async def chat_endpoint(payload: QueryPayload):
#     answer = rag_chain.invoke(payload.query)
    
#     retrieved_docs = retriever.invoke(payload.query)
#     sources = list(set([doc.metadata.get("source", "Internal Doc") for doc in retrieved_docs]))
    
#     return {
#         "text": answer,
#         "card": {
#             "title": "Sources Consulted",
#             "theme": "modern-inline",
#             "thumbnail": ""
#         },
#         "sources": sources
#     }

# from fastapi import FastAPI
# from pydantic import BaseModel
# from langchain_qdrant import QdrantVectorStore
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# app = FastAPI()

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# # Connect to the existing collection running in Docker
# vector_db = QdrantVectorStore.from_existing_collection(
#     embedding=embeddings,
#     url="http://localhost:6333",  # Local Docker Qdrant instance
#     collection_name="gitktek_knowledge_base",
#     vector_name="giktek-dense-vector"
# )

# retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# prompt = ChatPromptTemplate.from_template("""
# You are a helpful company assistant. Answer the employee's question using ONLY the context provided below.
# If you don't know the answer, state that you don't have this record in the company knowledge base.

# Context:
# {context}

# Question: {input}
# """)

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# document_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, document_chain)

# class QueryPayload(BaseModel):
#     query: str
#     user_id: str

# @app.post("/chat")
# async def chat_endpoint(payload: QueryPayload):
#     response = rag_chain.invoke({"input": payload.query})
    
#     sources = list(set([doc.metadata.get("source", "Internal Doc") for doc in response["context"]]))
    
#     return {
#         "text": response["answer"],
#         "card": {
#             "title": "Sources Consulted",
#             "theme": "modern-inline",
#             "thumbnail": ""
#         },
#         "sources": sources
#     }