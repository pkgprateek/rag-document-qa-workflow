# Test rag pipeline
from app.rag_pipeline import RAGPipeline
from app.document_processor import DocumentProcessor


processor = DocumentProcessor()
# chunks = processor.process_pdf("./data/test.pdf")
test_doc = processor.process_txt(
    """
    Python is a high-level programming language.
    It was created by Guido van Rossum in 1991.
    Python is known for its simple syntax., 
    test_python.txt
    """
)

# Initialize Rag
rag_pipeline = RAGPipeline()
rag_pipeline.add_documents(test_doc)

# Query
question = "What is python known for?"
result = rag_pipeline.query(question)
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} chunks retrieved.")
