# FYP Handbook RAG Assistant

A Retrieval-Augmented Generation (RAG) pipeline for answering questions about the FAST-NUCES Final Year Project Handbook 2023.

## Features

- 📄 **PDF Processing**: Extracts text from the handbook with page preservation
- 🔍 **Semantic Search**: Uses Sentence-BERT (all-MiniLM-L6-v2) for embeddings
- 💾 **FAISS Index**: Fast similarity search with local FAISS index
- 📚 **Page Citations**: Every answer includes page references
- 🎨 **Streamlit UI**: Clean, user-friendly interface

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Ingest the PDF

Run the ingestion script to process the PDF and create the FAISS index:

```bash
python ingest.py
```

This will:
- Extract text from `FYP-Handbook-2023.pdf`
- Chunk the text (300 words per chunk, 30% overlap)
- Create embeddings using all-MiniLM-L6-v2
- Build and save a FAISS index
- Store metadata in `faiss_index/` and `embeddings_data/`

### Step 2: Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser. You can then ask questions about the FYP handbook!

## Example Questions

1. "What headings, fonts, and sizes are required in the FYP report?"
2. "What margins and spacing do we use?"
3. "What are the required chapters of a Development FYP report?"
4. "What are the required chapters of an R&D-based FYP report?"
5. "How should endnotes like 'Ibid.' and 'op. cit.' be used?"
6. "What goes into the Executive Summary and Abstract?"

## Architecture

- **ingest.py**: PDF parsing, chunking, embedding, and FAISS index creation
- **app.py**: Streamlit interface with query processing and answer generation
- **prompt_log.txt**: System prompts and configuration details

## Configuration

- **Chunk Size**: 300 words (configurable in `ingest.py`)
- **Overlap**: 30% (configurable in `ingest.py`)
- **Top-K Retrieval**: 8 chunks
- **Similarity Threshold**: 0.18

## Files Structure

```
.
├── FYP-Handbook-2023.pdf    # Source PDF
├── ingest.py                 # Ingestion script
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── prompt_log.txt            # System prompts
├── faiss_index/             # Generated FAISS index
│   └── faiss_index.bin
└── embeddings_data/         # Generated metadata
    ├── chunks_metadata.json
    └── model_info.json
```

## Troubleshooting

### PyTorch/Streamlit Compatibility Warning

If you see warnings about `torch.classes` when running Streamlit, this is a known compatibility issue and can be safely ignored. The app includes:
- Warning filters in `app.py` to suppress these messages
- Environment variable + config to force the polling file watcher (`STREAMLIT_SERVER_FILEWATCHER_TYPE=poll`)

The app will function normally despite these warnings.

## OCR Support

The ingestion pipeline includes intelligent OCR fallback:
- **Primary**: Uses `pdfplumber` for text extraction
- **Fallback**: Automatically uses `pytesseract` OCR when:
  - Text extraction returns empty text
  - Extracted text is too short (< 30 characters)
- **Requirements**: 
  - **Tesseract OCR** must be installed on your system:
    - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
    - Linux: `sudo apt-get install tesseract-ocr`
    - macOS: `brew install tesseract`
  - **Poppler** (for pdf2image):
    - Windows: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/) and add to PATH
    - Linux: `sudo apt-get install poppler-utils`
    - macOS: `brew install poppler`
- OCR is only used when needed, keeping processing fast for text-based pages

## Notes

- The system uses extraction-based answer generation (no LLM required)
- All answers are grounded in the handbook with page citations
- Similarity threshold filters out irrelevant queries
- The index is persisted to disk for fast subsequent queries
- OCR fallback ensures complete text extraction from scanned pages and images

