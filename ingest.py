"""
FYP Handbook RAG Pipeline - Ingestion Script
Loads PDF, chunks text, creates embeddings, and builds FAISS index.
Includes OCR fallback for pages with scanned images or incomplete text extraction.
"""

import os
import json
from typing import List, Dict, Tuple
import re

import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# OCR dependencies (optional - only used as fallback)
PDF2IMAGE_AVAILABLE = False
PYTESSERACT_AVAILABLE = False

# Manual path configuration (set these if auto-detection fails)
# Uncomment and set the paths to your installation locations:
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.11.0\Library\bin"

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    pass  # Will be handled gracefully

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pass  # Will be handled gracefully


def configure_tesseract_path():
    """Configure Tesseract path, especially for Windows installations."""
    import platform
    
    if not PYTESSERACT_AVAILABLE:
        return False
    
    # Check if manual path is set
    if 'TESSERACT_PATH' in globals() and TESSERACT_PATH:
        if os.path.exists(TESSERACT_PATH):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
            try:
                pytesseract.get_tesseract_version()
                print(f"Using manually configured Tesseract at: {TESSERACT_PATH}")
                return True
            except Exception:
                pass
    
    # Try to get Tesseract version (this will fail if not in PATH)
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        # Tesseract not in PATH, try common Windows installation paths
        if platform.system() == 'Windows':
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    try:
                        pytesseract.get_tesseract_version()
                        print(f"Found Tesseract at: {path}")
                        return True
                    except Exception:
                        continue
        
        return False


def configure_poppler_path():
    """Configure Poppler path, especially for Windows installations."""
    import platform
    import shutil
    
    if not PDF2IMAGE_AVAILABLE:
        return False
    
    # Check if manual path is set
    if 'POPPLER_PATH' in globals() and POPPLER_PATH:
        if os.path.exists(POPPLER_PATH):
            poppler_exe = os.path.join(POPPLER_PATH, 'pdftoppm.exe')
            if os.path.exists(poppler_exe):
                os.environ['PATH'] = POPPLER_PATH + os.pathsep + os.environ.get('PATH', '')
                print(f"Using manually configured Poppler at: {POPPLER_PATH}")
                return True
    
    # Check if poppler is already in PATH
    if shutil.which('pdftoppm') or shutil.which('pdftocairo'):
        return True
    
    if platform.system() == 'Windows':
        # Common Poppler installation paths on Windows
        common_paths = [
            r'C:\poppler\poppler-25.11.0\Library\bin',
            r'C:\poppler\poppler-25.11.0\Library\bin',
            r'C:\poppler\poppler-25.11.0\Library\bin',
            r'C:\poppler\poppler-25.11.0\Library\bin'.format(os.getenv('USERNAME', '')),
        ]
        
        # Try to find poppler in common locations
        for path in common_paths:
            if os.path.exists(path):
                poppler_exe = os.path.join(path, 'pdftoppm.exe')
                if os.path.exists(poppler_exe):
                    # Set environment variable for pdf2image
                    os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
                    print(f"Found Poppler at: {path}")
                    return True
        
        return False
    
    return True  # On Linux/Mac, assume it's in PATH

# Configuration
CHUNK_SIZE = 300  # Target words per chunk (250-400 range)
CHUNK_OVERLAP = 0.3  # 30% overlap (20-40% range)
SIMILARITY_THRESHOLD = 0.25
TOP_K = 5
OCR_MIN_TEXT_LENGTH = 30  # Minimum characters to consider text extraction successful

# Paths
PDF_PATH = "FYP-Handbook-2023.pdf"
INDEX_DIR = "faiss_index"
EMBEDDINGS_DIR = "embeddings_data"

# Create directories
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


def clean_ocr_text(text: str) -> str:
    """
    Clean OCR text by removing common OCR artifacts and normalizing whitespace.
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR errors (optional - can be expanded)
    # Remove isolated characters that are likely OCR errors
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('—', '-').replace('–', '-')
    
    # Remove page numbers and headers/footers if they appear
    # (This is heuristic and may need adjustment)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip lines that are just numbers (likely page numbers)
        if line.isdigit() and len(line) < 4:
            continue
        # Skip very short lines that are likely artifacts
        if len(line) < 3:
            continue
        cleaned_lines.append(line)
    
    text = ' '.join(cleaned_lines)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_text_with_ocr_fallback(pdf_path: str, page_num: int, ocr_enabled: bool = True) -> str:
    """
    Extract text from a single PDF page using OCR as fallback.
    Returns extracted text (from pdfplumber or OCR).
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num > len(pdf.pages):
                return ""
            
            page = pdf.pages[page_num - 1]  # pdfplumber uses 0-based indexing
            text = page.extract_text()
            
            # Clean up text
            if text:
                text = re.sub(r'\s+', ' ', text).strip()
            
            # Check if text extraction was successful
            # If text is empty or too short, use OCR (if available and enabled)
            if (not text or len(text) < OCR_MIN_TEXT_LENGTH) and ocr_enabled and PDF2IMAGE_AVAILABLE and PYTESSERACT_AVAILABLE:
                print(f"  Page {page_num}: Text extraction insufficient ({len(text) if text else 0} chars), using OCR...")
                
                # Convert PDF page to image
                try:
                    images = convert_from_path(
                        pdf_path,
                        first_page=page_num,
                        last_page=page_num,
                        dpi=300  # Higher DPI for better OCR accuracy
                    )
                    
                    if images:
                        # Run OCR on the image
                        ocr_text = pytesseract.image_to_string(images[0], lang='eng')
                        ocr_text = clean_ocr_text(ocr_text)
                        
                        if ocr_text and len(ocr_text) >= OCR_MIN_TEXT_LENGTH:
                            print(f"  Page {page_num}: OCR successful ({len(ocr_text)} chars)")
                            return ocr_text
                        else:
                            print(f"  Page {page_num}: OCR returned insufficient text ({len(ocr_text) if ocr_text else 0} chars)")
                            return text if text else ""  # Return original text or empty
                    else:
                        print(f"  Page {page_num}: Failed to convert page to image")
                        return text if text else ""
                        
                except Exception as ocr_error:
                    print(f"  Page {page_num}: OCR failed: {ocr_error}")
                    return text if text else ""  # Fallback to original text
            elif not text or len(text) < OCR_MIN_TEXT_LENGTH:
                # OCR not available, but text is insufficient
                print(f"  Page {page_num}: Text extraction insufficient ({len(text) if text else 0} chars), OCR not available")
                return text if text else ""
            
            return text
            
    except Exception as e:
        print(f"  Page {page_num}: Error extracting text: {e}")
        return ""


def extract_text_with_pages(pdf_path: str, ocr_enabled: bool = True) -> List[Tuple[int, str]]:
    """
    Extract text from PDF preserving page numbers.
    Uses OCR as fallback when text extraction fails or returns insufficient text.
    Returns list of (page_number, text) tuples.
    """
    pages_data = []
    
    print(f"Extracting text from PDF: {pdf_path}")
    
    # First, get total number of pages
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return pages_data
    
    # Extract text from each page
    for page_num in range(1, total_pages + 1):
        text = extract_text_with_ocr_fallback(pdf_path, page_num, ocr_enabled)
        if text:  # Only add pages with text
            pages_data.append((page_num, text))
        else:
            print(f"  Page {page_num}: No text extracted (skipping)")
    
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


def process_pdf(pdf_path: str, ocr_enabled: bool = True) -> List[Dict]:
    """
    Process PDF: extract text, chunk, and create metadata.
    Returns list of chunk dictionaries with metadata.
    """
    print(f"Loading PDF: {pdf_path}")
    pages_data = extract_text_with_pages(pdf_path, ocr_enabled)
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
    
    # Check and configure OCR dependencies
    tesseract_configured = False
    poppler_configured = False
    ocr_enabled = False
    
    if PYTESSERACT_AVAILABLE:
        tesseract_configured = configure_tesseract_path()
        if not tesseract_configured:
            print("Warning: Tesseract OCR not found in PATH or common installation locations.")
            print("  Please ensure Tesseract is installed and either:")
            print("  1. Add it to your system PATH, or")
            print("  2. Set pytesseract.pytesseract.tesseract_cmd in the script")
    
    if PDF2IMAGE_AVAILABLE:
        poppler_configured = configure_poppler_path()
        if not poppler_configured:
            print("Warning: Poppler not found in PATH or common installation locations.")
            print("  Please ensure Poppler is installed and either:")
            print("  1. Add it to your system PATH, or")
            print("  2. Set the PATH environment variable to include Poppler's bin directory")
    
    ocr_enabled = (PDF2IMAGE_AVAILABLE and PYTESSERACT_AVAILABLE and 
                   tesseract_configured and poppler_configured)
    
    if ocr_enabled:
        print("OCR support: Available and configured (pytesseract + pdf2image)")
    elif PDF2IMAGE_AVAILABLE and PYTESSERACT_AVAILABLE:
        print("OCR support: Dependencies installed but configuration incomplete")
        print("Continuing with text extraction only (OCR will be skipped)...")
    else:
        print("OCR support: Not available (missing dependencies)")
        print("Continuing with text extraction only...")
    
    # Step 1: Load and chunk PDF
    chunks = process_pdf(PDF_PATH, ocr_enabled)
    
    if not chunks:
        print("Error: No chunks created. Please check the PDF file.")
        return
    
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
