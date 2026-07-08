import os
import pandas as pd
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List

from paths import FAISS_INDEX, METADATA_CSV, TEXTBOOK_DIR

# === CONFIG ===
PDF_DIR = str(TEXTBOOK_DIR) + os.sep
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 3000
FAISS_INDEX_PATH = str(FAISS_INDEX)
METADATA_PATH = str(METADATA_CSV)

print("🔧 Configuration loaded.")
print(f"PDF directory: {PDF_DIR}")
print(f"Model: {MODEL_NAME}")
print(f"Chunk size: {CHUNK_SIZE}\n")

# === FUNCTIONS ===
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        return ""

def chunk_text(text: str, max_length: int = 1000) -> List[str]:
    chunks = []
    while len(text) > max_length:
        split_idx = text.rfind('.', 0, max_length)
        split_idx = split_idx if split_idx != -1 else max_length
        chunks.append(text[:split_idx+1].strip())
        text = text[split_idx+1:].strip()
    if text:
        chunks.append(text)
    return chunks

# === LOAD MODEL ===
print("📦 Loading model...")
model = SentenceTransformer(MODEL_NAME)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"✅ Model loaded: Embedding dimension = {embedding_dim}")

# === SCAN TEXTBOOK FOLDER FOR PDFs ===
# (previously this read a curated CSV's 'file_path' column; now we just
#  scan the folder directly so newly added PDFs are picked up automatically)
print("\n📂 Scanning textbook folder for PDFs...")
pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
file_paths = [os.path.splitext(f)[0] for f in pdf_files]
print(f"📚 Found {len(file_paths)} PDF files.\n")

# === INIT FAISS ===
index = faiss.IndexFlatL2(embedding_dim)
metadata = []

# === PROCESS PDFs ===
for file_name in file_paths:
    pdf_path = os.path.join(PDF_DIR, f"{file_name}.pdf")

    if not os.path.isfile(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        continue

    print(f"\n📄 Processing PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        print("⚠️ No text extracted from PDF. Skipping.")
        continue

    chunks = chunk_text(text, max_length=CHUNK_SIZE)
    print(f"✂️  Chunked into {len(chunks)} segments.")

    try:
        embeddings = model.encode(chunks, show_progress_bar=True)
    except Exception as e:
        print(f"❌ Embedding failed for {pdf_path}: {e}")
        continue

    index.add(np.array(embeddings).astype("float32"))
    metadata.extend([{"source": file_name, "text": chunk} for chunk in chunks])
    print(f"📌 Added {len(chunks)} embeddings to FAISS.")

# === SAVE FAISS INDEX AND METADATA ===
print("\n💾 Saving FAISS index and metadata...")
faiss.write_index(index, FAISS_INDEX_PATH)
pd.DataFrame(metadata).to_csv(
    METADATA_PATH,
    index=False,
    escapechar='\\',
    quoting=1  # csv.QUOTE_ALL
)
print(f"\n✅ FAISS index saved to: {FAISS_INDEX_PATH}")
print(f"✅ Metadata saved to: {METADATA_PATH}")
