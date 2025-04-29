# RAG-ecosystem

A complete Retrieval-Augmented Generation (RAG) system, plus an OCR-to-Markdown & Image Q&A module, built with Streamlit and Haystack.

## Table of Contents

- [Features](#features)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Configuration](#configuration)  
- [Usage](#usage)  
  - [1. Agentic RAG Chat & BM25 Search](#1-agentic-rag-chat---bm25-search)  
  - [2. OCR to Markdown & Image Q&A](#2-ocr-to-markdown---image-qa)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  
- [License](#license)

## Project Structure 
```
.
├── RAG.py                 # Agentic RAG & BM25 Search Streamlit app
├── ocr_processor.py       # OCR → Markdown & Image Q&A Streamlit app
├── utils/
│   └── custom_converters.py  # Docx/.xlsx → Haystack Document converters
├── chat_history.db        # SQLite DB (auto-generated)
├── LICENSE                # MIT License
└── .gitignore

```

## Features

1. **Agentic RAG Chat**  
   - Hybrid retrieval (OpenAI embeddings + BM25)  
   - Contextual query routing & summarization  
   - Stateful chat sessions saved to SQLite  

2. **BM25-Only Document Search**  
   - Quick keyword-driven document lookup  

3. **OCR to Markdown Converter**  
   - Uses Together AI vision models to extract full-page content as Markdown  

4. **Image Question-Answering**  
   - Ask natural-language questions about any uploaded or URL’d image  

## Getting Started

### Prerequisites

- **Python 3.8+**  
- An **OpenAI** API key  
- A **Together AI** API key (for OCR & Image Q&A)  
- Git & your favorite terminal/shell  

### Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Esmail-ibraheem/RAG-ecosystem.git
   cd RAG-ecosystem
