An AI-powered HR Policy Assistant that answers employee queries directly from company HR documents using Retrieval-Augmented Generation (RAG).

👉 No hardcoded responses.

👉 No hallucinations.

👉 Answers strictly from HR policy documents.

🎯 Problem Statement

## **Employees often struggle with:**

Searching through hundreds of HR policy pages

Waiting for HR responses

Misinterpreting policy rules

This project solves that by building an AI assistant that reads HR documents and answers instantly.

## 🧠 **How It Works**

User Query
   ↓
Query Embedding
   ↓
Vector Search (Milvus)
   ↓
Relevant Policy Chunks
   ↓
LLM (Groq - LLaMA 3.3)
   ↓
Final Answer

## **Tech Stack**

Language: Python

RAG Framework: LlamaIndex

Embeddings: SentenceTransformers (all-mpnet-base-v2)

Vector Database: Milvus

LLM Inference: Groq (LLaMA 3.3 70B)

## ⚙️ **Features**

📄 Load HR policy documents (.docx)

✂️ Intelligent chunking with overlap

🔍 Semantic search using embeddings

⚡ Fast retrieval using Milvus

🧠 Context-aware response generation

✅ Strictly grounded answers (no hallucination)

## 🔄 **Pipeline Breakdown**

### 1. **Document Loading**

Uses SimpleDirectoryReader

Supports .docx files via DocxReader

### 2. **Chunking**

Chunk size: 512 tokens

Overlap: 100 tokens

Ensures better context retention

### 3. **Embedding Generation**

Model: all-mpnet-base-v2

Output: 768-dimensional vectors

### 4. **Vector Storage**

Database: Milvus

Index Type: AUTOINDEX

Similarity: Cosine

### 5. **Retrieval**

Top-K search (default: 5)

Returns most relevant policy chunks

### 6. **Response Generation**

LLM: Groq (LLaMA 3.3 70B)

Prompt ensures:

Professional tone

Context-only answers
