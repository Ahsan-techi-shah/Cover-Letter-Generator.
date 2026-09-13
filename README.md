---
title: AI Job Cover Letter Generator
description: RAG-based cover letter generator using Streamlit, FAISS, Sentence Transformers and Groq.
tags:
  - streamlit
  - rag
  - groq
  - faiss
  - sentence-transformers
  - cover-letter
sdk: docker
app_port: 8501
---

# AI Job Cover Letter Generator

A Retrieval-Augmented Generation (RAG) Streamlit app that creates a tailored cover letter from a candidate CV and a job advertisement.

## Features

- CV upload: PDF, JPG, JPEG, PNG
- Job ad upload: PDF, JPG, JPEG, PNG
- PDF text extraction with OCR fallback for scanned PDFs
- Image OCR using Tesseract
- Sentence Transformer embeddings
- FAISS similarity search
- Relevant CV/job-ad passage retrieval
- Groq-powered cover letter generation
- Download generated cover letter as a TXT file

## Hugging Face Spaces setup

1. Create a new Hugging Face Space.
2. Choose the **Docker** SDK and Streamlit template. Hugging Face currently recommends Docker for Streamlit Spaces. 
3. Upload all files in this repository.
4. Open **Settings → Secrets** and add:

```text
GROQ_API_KEY = your_groq_api_key
```

Do not put the API key inside `app.py` or commit it to GitHub.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tesseract must also be installed on the operating system for OCR.
