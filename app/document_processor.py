from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import PyPDF2
from docx import Document as DocxDocument


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def process_pdf(self, file_path: str) -> List[Document]:
        """Extract text from a PDF file and split it into chunks"""
        reader = PyPDF2.PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n---- Page {page_num + 1} ----\n{page_text}"
        # Create documents with metadata
        chunks = self.text_splitter.create_documents(
            [text],
            metadatas=[{"source": file_path, "type": "pdf"}],
        )
        return chunks

    def process_docx(self, file_path: str) -> List[Document]:
        """Extract text from a DOCX file and split it into chunks"""
        doc = DocxDocument(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        # Create documents with metadata
        chunks = self.text_splitter.create_documents(
            [text],
            metadatas=[{"source": file_path, "type": "docx"}],
        )
        return chunks

    def process_txt(self, text: str, source: str = "user_input") -> List[Document]:
        """Process raw text into chunks"""
        chunks = self.text_splitter.create_documents(
            [text],
            metadatas=[{"source": source, "type": "txt"}],
        )
        return chunks
    