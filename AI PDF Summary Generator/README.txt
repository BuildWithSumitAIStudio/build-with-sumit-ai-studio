AI PDF Summary Generator (Groq + Python) :-
Generate high-quality summaries from large PDF files using AI. This project uses Groq LLMs to process long documents efficiently by splitting them into chunks and summarizing them step-by-step.

Steps :-
Extract text from large PDFs
Smart chunking (sentence-aware splitting)
AI-powered summarization (chunk-wise)
Final structured summary generation
Fast inference using Groq models

Architecture :-
PDF → Text Extraction → Chunking → AI Summaries → Final Summary

Tech Stack :-
Python 
Groq API (LLM) 
PyPDF2

Installation :-
pip install groq PyPDF2

Setup API Key :-
MAC/Linux -> export groq_api_key="your_api_key_here"
Windows -> set groq_api_key=your_api_key_here

Code Overview :-
1. Extract Text from PDF
   extract_text_from_pdf(pdf_path)
2. Smart Chunking
   chunk_text(text)
3. Chunk-wise Summarization
   summarize_chunk(chunk)
4. Final Summary Generation
   generate_final_summary(all_summaries)
