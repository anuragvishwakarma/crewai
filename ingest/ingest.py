# ingest/ingest.py
import os
import pandas as pd
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

DATA_DIR = Path("/Users/koki/Documents/testing/Rag/data")
OUTPUT_DIR = Path("/Users/koki/Documents/testing/Rag/models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- PDF LOADER ---
def load_pdfs():
    docs = []
    manuals_dir = DATA_DIR / "manuals"
    if not manuals_dir.exists():
        print("⚠️ No 'manuals' folder found.")
        return docs
        
    for pdf_file in manuals_dir.glob("*.pdf"):
        print(f"📄 Loading: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            for page in pages:
                page.metadata.update({"source": "manual", "file": pdf_file.name})
            docs.extend(pages)
        except Exception as e:
            print(f"❌ PDF error ({pdf_file.name}): {e}")
    return docs

# --- SAFE CSV LOADER ---
def load_maintenance_csv():
    csv_path = DATA_DIR / "synthetic_maintenance_records.csv"
    if not csv_path.exists():
        print("⚠️ CSV file not found.")
        return []

    docs = []
    expected_columns = ["equipment_id", "date", "component_code", "component_description", "inspection_type", "technician", "notes"]
    
    with open(csv_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(";")
            
            # Pad or truncate to exactly 7 fields
            while len(parts) < 7:
                parts.append("")  # Add empty string for missing fields
            if len(parts) > 7:
                parts = parts[:7]  # Keep only first 7
                
            try:
                row = dict(zip(expected_columns, parts))
                content = (
                    f"[MAINTENANCE RECORD]\n"
                    f"Equipment: {row['equipment_id']}\n"
                    f"Date: {row['date']}\n"
                    f"Component ({row['component_code']}): {row['component_description']}\n"
                    f"Technician: {row['technician']}\n"
                    f"Notes: {row['notes']}"
                )
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": "maintenance_record",
                        "equipment_id": row["equipment_id"],
                        "component_code": row["component_code"]
                    }
                ))
            except Exception as e:
                print(f"❌ Skipping line {line_num}: {e}")
                continue
                
    print(f"✅ Loaded {len(docs)} records from CSV")
    return docs

# --- MAIN INGESTION ---
def main():
    all_docs = load_pdfs() + load_maintenance_csv()
    if not all_docs:
        print("❗ No documents loaded. Check your data files.")
        return

    print(f"✅ Total documents: {len(all_docs)}")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    print(f"🔀 Created {len(chunks)} chunks")

    # Embed & FAISS
    print("🧠 Generating embeddings with Ollama (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(OUTPUT_DIR))
    print(f"💾 FAISS index saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()