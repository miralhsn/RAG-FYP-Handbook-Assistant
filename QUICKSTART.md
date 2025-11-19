# Quick Start Guide

## Setup (One-time)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run ingestion (processes the PDF and creates the index):**
   ```bash
   python ingest.py
   ```
   
   This will:
   - Extract text from `FYP-Handbook-2023.pdf`
   - Create ~300-word chunks with 30% overlap
   - Generate embeddings using all-MiniLM-L6-v2
   - Build FAISS index
   - Save everything to `faiss_index/` and `embeddings_data/`
   
   **Note:** First run will download the embedding model (~80MB). This is a one-time download.

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
- **Torch watcher warning**: The app now forces Streamlit to use the `poll` file watcher and suppresses the `torch.classes` warning automatically. If you still see it, restart the Streamlit app.

