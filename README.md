<h1 align="center">📚 FYP Handbook RAG Assistant</h1>

<p align="center">
  <b>Retrieval-Augmented Generation System for FAST-NUCES FYP Handbook</b>
</p>

<p align="center">
  <a href="https://YOUR-STREAMLIT-LINK.streamlit.app">
    🚀 Live Demo
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/RAG-Semantic%20Search-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20DB-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SentenceBERT-Embeddings-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge"/>
</p>

---

## 🧠 Overview

This system is a **Retrieval-Augmented Generation (RAG) pipeline** designed to answer questions from the **FAST-NUCES FYP Handbook (2023)**.

Instead of manually searching through PDFs, users can ask questions and get:

- 🎯 Context-aware answers  
- 📄 Page-level citations  
- ⚡ Sub-second semantic retrieval  

---

## 🚀 Live Demo

<p align="center">
  <img src="assets/demo.gif" width="800"/>
</p>

> Replace this with a screen recording of your Streamlit app

---

## ⚙️ System Architecture

PDF → Chunking → Sentence-BERT Embeddings → FAISS Index → Semantic Retrieval → Answer Output


---

## 🔑 Key Features

- 📄 PDF parsing with page preservation  
- 🔍 Semantic search using Sentence-BERT  
- 💾 FAISS vector similarity search  
- 📚 Page-level citation tracking  
- 🎨 Streamlit interactive UI  
- ⚡ Fast local retrieval (no LLM required)  
- 🧠 OCR fallback for scanned documents  

---

## 🧠 Problem Statement

Students struggle with:

- Searching long FYP guidelines manually  
- Missing important formatting rules  
- Slow navigation through PDF documents  

---

## 💡 Solution

This system transforms static PDFs into an **intelligent Q&A system**:

- Ask natural language questions  
- Get exact extracted answers  
- Receive page citations for verification  

---

## 🛠️ Tech Stack

- **Embeddings:** Sentence-BERT (all-MiniLM-L6-v2)  
- **Vector DB:** FAISS  
- **Backend:** Python  
- **UI:** Streamlit  
- **OCR:** Tesseract (fallback system)  

---

## 📊 Configuration

| Parameter | Value |
|----------|------|
| Chunk Size | 300 words |
| Overlap | 30% |
| Top-K Retrieval | 8 chunks |
| Similarity Threshold | 0.18 |

---

## 🧩 Engineering Highlights

- Optimized chunking for semantic coherence  
- FAISS indexing for fast retrieval  
- Hybrid text + OCR pipeline  
- Cached embeddings for performance  

---

## 🔮 Future Improvements

- Integration with LLMs for answer refinement  
- Multi-document RAG system  
- Cloud-based vector database  
- Query history + analytics dashboard  

---

## 📁 Project Structure
├── app.py
├── ingest.py
├── FYP-Handbook-2023.pdf
├── faiss_index/
├── embeddings_data/
└── requirements.txt



---

## 🧠 Why This Project Matters

This project demonstrates:

- Real-world RAG pipeline design  
- Vector database usage (FAISS)  
- NLP embedding systems  
- Production-style AI application design  
- End-to-end AI system building  

---

## 👨‍💻 Author

**Miral Hasan**

<p align="center">
  <a href="https://github.com/miralhsn">GitHub</a> •
  <a href="https://linkedin.com/in/miral-hasan-26353b249">LinkedIn</a>
</p>

---

<p align="center">
  ⭐ If you like this project, consider giving it a star!
</p>
