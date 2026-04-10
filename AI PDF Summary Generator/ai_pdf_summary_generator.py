

import os
from groq import Groq
from PyPDF2 import PdfReader

# =========================
# CONFIG
# =========================
GROQ_API_KEY = os.getenv("groq_api_key")
client = Groq(api_key=GROQ_API_KEY)

# =========================
# STEP 1: READ PDF
# =========================
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text.append(text)

    return "\n".join(full_text)

# =========================
# STEP 2: SMART CHUNKING
# =========================
def chunk_text(text: str, max_chars: int = 4000):
    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        # Try to split at sentence boundary
        if end < len(text):
            last_period = (
                text.rfind(".", start, end))
            if last_period != -1:
                end = last_period + 1

        chunks.append(text[start:end].strip())
        start = end

    return chunks

# =========================
# STEP 3: AI SUMMARIZATION
# =========================
def summarize_chunk(chunk: str, index: int, total: int) -> str:
    print(f"Processing chunk {index+1}/{total}")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[ #type: ignore
            {
                "role": "system",
                "content": (
                    "You are an expert summarizer. "
                    "Summarize the text "
                    "into structured bullet points. "
                    "Focus on key insights, "
                    "avoid repetition."
                    "Generate summary within 100 words."
                ),
            },
            {
                "role": "user",
                "content": chunk
            }
        ]
    )

    return response.choices[0].message.content

# =========================
# STEP 4: FINAL SUMMARY
# =========================
def generate_final_summary(all_summaries: list) \
        -> str:
    combined = "\n".join(all_summaries)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[ # type: ignore
            {
                "role": "system",
                "content": (
                    "Create a final clean "
                    "summary with:\n"
                    "1. Key Points\n"
                    "2. Important Insights\n"
                    "3. Actionable Takeaways\n"
                ),
            },
            {
                "role": "user",
                "content": combined
            }
        ]
    )

    return response.choices[0].message.content

# =========================
# MAIN PIPELINE
# =========================
def summarize_pdf(pdf_path: str):
    print("Reading PDF...")
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        raise ValueError("No text found in PDF")

    print("Splitting into chunks...")
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    summaries = []
    for i, chunk in enumerate(chunks):
        summary = summarize_chunk(chunk, i, len(chunks))
        summaries.append(summary)

    print("Generating final summary...")
    final_output = generate_final_summary(summaries)

    return final_output


# =========================
# EXECUTION
# =========================
if __name__ == "__main__":
    pdf_file = "large_text_document.pdf"  # Replace with your file

    result = summarize_pdf(pdf_file)

    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50 + "\n")
    print(result)