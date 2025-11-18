from app.document_processor import DocumentProcessor

processor = DocumentProcessor()

pdf_path = "data/test.pdf"
chunks = processor.process_pdf(pdf_path)

print(f"Created {len(chunks)} chunks")
print(f"First chunk: {chunks[0].page_content[:100]}...")
print(f"Metadata: {chunks[0].metadata}")