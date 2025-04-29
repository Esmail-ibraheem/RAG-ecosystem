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
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install streamlit
   pip install "haystack[all]"   # Core RAG components
   pip install openai sqlalchemy pandas python-docx pillow requests python-dotenv
   ```

### Configuration

Create a `.env` file in the project root with your API keys:

```dotenv
OPENAI_API_KEY=sk-…
TOGETHER_API_KEY=sk-…
```

Or set them in your shell:

```bash
export OPENAI_API_KEY=sk-…
export TOGETHER_API_KEY=sk-…
```

## Usage

### 1. Agentic RAG Chat & BM25 Search

Launch the main RAG app:

```bash
streamlit run RAG.py
```

- **Sidebar**  
  - Enter your OpenAI API key.  
  - Pick a GPT model (e.g. `gpt-3.5-turbo` or `gpt-4-turbo`).  
  - Upload documents (`.pdf`, `.docx`, `.txt`, `.csv`, `.xlsx`) for RAG or BM25.  
  - Start or load chat sessions.

- **Main panel**  
  - For “RAG Chat”: chat with the system—behind the scenes it chooses between summary, context-driven answer, or simple reply.  
  - For “BM25 Search”: run keyword searches and preview top‐k results.

All chat history is stored in `chat_history.db` (SQLite) for later reuse.

### 2. OCR to Markdown & Image Q&A

Run the OCR & QA utility:

```bash
streamlit run ocr_processor.py
```

- **Convert to Markdown**  
  - Upload or URL-point to an image.  
  - Click **Convert to Markdown** to get full‐page Markdown.  

- **Ask About the Image**  
  - Enter a natural-language question.  
  - Click **Get Answer** to see the model’s response.

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

## Contributing

1. Fork the repo  
2. Create a feature branch (`git checkout -b feat/my-feature`)  
3. Commit your changes (`git commit -m "Add …"`)  
4. Push to your branch (`git push`)  
5. Open a Pull Request!

Please follow the existing code style and include tests/examples where applicable.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.  
```
