PDF-to-AI RAG Pipeline
Convert any large PDF into an intelligent, queryable AI assistant using a Retrieval-Augmented Generation (RAG)pipeline. This project demonstrates how to build a scalable, production-style system for document understanding + question answering using modern AI tooling.

Overview
This pipeline takes a large PDF document, processes it intelligently, stores it in a vector database, and enables natural language querying with highly relevant answers.
Flow: PDF → Semantic Chunking → Embeddings → Vector DB → LLM → Answer

Tech Stack
* LlamaIndex – Orchestration framework for RAG
* Milvus – High-performance vector database
* HuggingFace Embeddings – all-MiniLM-L6-v2
* Groq (LLaMA 3) – Fast LLM inference
* Python – End-to-end implementation

Key Features
* Semantic document splitting (context-aware chunking)
* Efficient embedding generation
* Scalable vector storage with Milvus
* Fast and accurate retrieval
* LLM-powered natural language responses
* Production-style modular pipeline

Pipeline Breakdown
1. Document Ingestion
Reads PDF files using SimpleDirectoryReader.
2. Semantic Chunking
Uses SemanticSplitterNodeParser to split content based on meaning, not just size.
3. Embedding Generation
Converts text chunks into vectors using HuggingFace MiniLM.
4. Vector Storage
Stores embeddings in Milvus for fast similarity search.
5. Index Creation
Builds a VectorStoreIndex for retrieval.
6. Query Engine
Transforms the index into a Q&A system using Groq LLaMA 3.
7. User Query
Accepts natural language input and returns contextual answers.
