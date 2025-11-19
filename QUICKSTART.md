# Quick Start Guide

## Setup (One-time)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note:** For OCR support (automatic fallback for scanned pages), you also need:
   - **Tesseract OCR**:
     - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
     - Linux: `sudo apt-get install tesseract-ocr`
     - macOS: `brew install tesseract`
   - **Poppler** (for PDF to image conversion):
     - Windows: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/) and add to PATH
     - Linux: `sudo apt-get install poppler-utils`
     - macOS: `brew install poppler`
   
   The pipeline will work without these, but OCR fallback won't be available for scanned pages.

2. **Run ingestion (processes the PDF and creates the index):**
   ```bash
   python ingest.py
   ```
   
   This will:
   - Extract text from `FYP-Handbook-2023.pdf` (using pdfplumber)
   - Automatically use OCR for pages with scanned images or incomplete text
   - Create ~300-word chunks with 30% overlap
   - Generate embeddings using all-MiniLM-L6-v2
   - Build FAISS index
   - Save everything to `faiss_index/` and `embeddings_data/`
   
   **Note:** 
   - First run will download the embedding model (~80MB). This is a one-time download.
   - Pages requiring OCR will be processed automatically (may take longer).

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Using the App

1. Enter your question in the text box
2. Click "Ask" button
3. View the answer with page citations
4. Expand "Sources" to see retrieved chunks
5. Use example questions from the sidebar for testing

## Validation Questions

Test these questions to validate the pipeline:

1. "What headings, fonts, and sizes are required in the FYP report?"
2. "What margins and spacing do we use?"
3. "What are the required chapters of a Development FYP report?"
4. "What are the required chapters of an R&D-based FYP report?"
5. "How should endnotes like 'Ibid.' and 'op. cit.' be used?"
6. "What goes into the Executive Summary and Abstract?"

## Troubleshooting

- **"Index files not found"**: Run `python ingest.py` first
- **Model download issues**: Check internet connection (first run only)
- **PDF not found**: Ensure `FYP-Handbook-2023.pdf` is in the project root
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
- **OCR errors**: If you see OCR-related errors, install Tesseract OCR (see step 1). The pipeline will continue without OCR, but scanned pages won't be processed.
- **Torch watcher warning**: The app now forces Streamlit to use the `poll` file watcher and suppresses the `torch.classes` warning automatically. If you still see it, restart the Streamlit app.

