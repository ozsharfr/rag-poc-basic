# Local Hebrew RAG – POC

Minimal offline RAG pipeline for querying Hebrew tax DOCX documents.

## What it does
• Split DOCX by headings  
• Chunk with overlap  
• Embed with multilingual‑E5  
• Store in Chroma  
• Hybrid retrieval (Vector + BM25)  
• Answer with local Ollama LLM  

Fully local. No cloud.

## Requirements
Python 3.10+

pip install:
torch chromadb langchain langchain-chroma langchain-huggingface rank-bm25 python-docx requests

## Setup
Install Ollama:
https://ollama.ai

Run:
ollama serve  
ollama pull gpt-oss:20b

Place:
DOCX → files_misui/

## Run
python rag_poc.py

## Key configs (inside code)
MAX_CHUNK_SIZE / OVERLAP – chunking  
RETRIEVE_MODE – vector | bm25 | hybrid  
EXPAND_QUERY = True – query expansion  
MODEL_NAME – Ollama model  

## Notes
• Works offline  
• Optimized for Hebrew legal text  
• Hybrid mode recommended  
