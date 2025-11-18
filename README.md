---
title: AI Document Intelligence System (with RAG)
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app/main.py
pinned: false
---

# AI Document Intelligence System
Upload documents and ask questions. Built with:
- LangChain for RAG orchestration
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Gradio for UI

## Features
- Interactive document processing
- Context-aware question answering
- Support for multiple file formats
- Real-time processing and analysis
- Multi-language support
- Customizable knowledge base

## Installation
To get started with the AI Document Intelligence System, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-document-intelligence.git
   cd ai-document-intelligence
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app/main.py
   ```
5. Open your web browser and navigate to the provided local URL to access the Gradio interface.

## Usage:
1. Upload a PDF/DOCX/TXT file
2. Click "Process Document"
3. Ask questions about the content
4. Get answers with source citations