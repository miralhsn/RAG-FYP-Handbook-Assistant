"""
FYP Handbook RAG Pipeline - Ingestion Script
Loads PDF, chunks text, creates embeddings, and builds FAISS index.
"""

import os
import json
import pickle
from typing import List, Dict, Tuple
import re

import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Configuration
CHUNK_SIZE = 300  # Target words per chunk (250-400 range)
CHUNK_OVERLAP = 0.3  # 30% overlap (20-40% range)
SIMILARITY_THRESHOLD = 0.25
TOP_K = 5

# Paths
PDF_PATH = "FYP-Handbook-2023.pdf"
INDEX_DIR = "faiss_index"
EMBEDDINGS_DIR = "embeddings_data"

# Create directories
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


def extract_text_with_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extract text from PDF preserving page numbers.
    Returns list of (page_number, text) tuples.
    """
    pages_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                # Clean up text
                text = re.sub(r'\s+', ' ', text).strip()
                pages_data.append((page_num, text))
    
    return pages_data


def find_first_heading(text: str) -> str:
    """
    Find the first heading in the text chunk.
    Looks for patterns like "Chapter X", "Section X", or numbered headings.
    """
    # Patterns for headings
    heading_patterns = [
        r'^Chapter\s+\d+[\.:]?\s*[A-Z][^\n]*',
        r'^Section\s+\d+[\.:]?\s*[A-Z][^\n]*',
        r'^\d+\.\d+\s+[A-Z][^\n]*',  # Numbered sections like "1.1 Title"
        r'^[A-Z][A-Z\s]{10,}',  # ALL CAPS headings
    ]
    
    lines = text.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if len(line) > 5:  # Minimum heading length
            for pattern in heading_patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    return line[:100]  # Limit heading length
    
    return "Introduction"  # Default


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: float = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks with overlap.
    chunk_size: target number of words
    overlap: fraction of chunk to overlap (0.2 = 20%)
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    step = max(1, int(chunk_size * (1 - overlap)))  # Ensure step is at least 1
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_content = ' '.join(chunk_words)
        if chunk_content.strip():  # Only add non-empty chunks
            chunks.append(chunk_content)
        
        if end >= len(words):
            break
        start += step
    
    return chunks


def process_pdf(pdf_path: str) -> List[Dict]:
    """
    Process PDF: extract text, chunk, and create metadata.
    Returns list of chunk dictionaries with metadata.
    """
    print(f"Loading PDF: {pdf_path}")
    pages_data = extract_text_with_pages(pdf_path)
    print(f"Extracted {len(pages_data)} pages")
    
    all_chunks = []
    chunk_id = 0
    
    for page_num, page_text in pages_data:
        # Chunk the page text
        chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        for chunk_content in chunks:
            section_hint = find_first_heading(chunk_content)
            
            chunk_data = {
                'chunk_id': chunk_id,
                'page': page_num,
                'section_hint': section_hint,
                'text': chunk_content,
                'word_count': len(chunk_content.split())
            }
            
            all_chunks.append(chunk_data)
            chunk_id += 1
    
    print(f"Created {len(all_chunks)} chunks")
    return all_chunks


def create_embeddings_and_index(chunks: List[Dict]) -> Tuple[faiss.Index, SentenceTransformer]:
    """
    Create embeddings for chunks and build FAISS index.
    Returns FAISS index and the embedding model.
    """
    print("Loading embedding model: all-MiniLM-L6-v2")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Creating embeddings...")
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Embedding shape: {embeddings.shape}")
    
    # Create FAISS index (L2 distance, but we'll use cosine similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    print(f"FAISS index created with {index.ntotal} vectors")
    
    return index, model


def save_index_and_metadata(index: faiss.Index, chunks: List[Dict], model: SentenceTransformer):
    """
    Save FAISS index and chunk metadata to disk.
    """
    # Save FAISS index
    index_path = os.path.join(INDEX_DIR, "faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")
    
    # Save chunk metadata
    metadata_path = os.path.join(EMBEDDINGS_DIR, "chunks_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {metadata_path}")
    
    # Save model info (model will be reloaded from sentence-transformers)
    model_info = {
        'model_name': 'all-MiniLM-L6-v2',
        'dimension': index.d,
        'num_chunks': len(chunks)
    }
    model_info_path = os.path.join(EMBEDDINGS_DIR, "model_info.json")
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"Saved model info to {model_info_path}")


def main():
    """
    Main ingestion pipeline.
    """
    print("=" * 60)
    print("FYP Handbook RAG Pipeline - Ingestion")
    print("=" * 60)
    
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}")
        return
    
    # Step 1: Load and chunk PDF
    chunks = process_pdf(PDF_PATH)
    
    # Step 2: Create embeddings and FAISS index
    index, model = create_embeddings_and_index(chunks)
    
    # Step 3: Save index and metadata
    save_index_and_metadata(index, chunks, model)
    
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"Total chunks: {len(chunks)}")
    print(f"Index dimension: {index.d}")
    print(f"Index size: {index.ntotal}")
    print("=" * 60)


if __name__ == "__main__":
    main()

