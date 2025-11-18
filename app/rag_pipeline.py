from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List

class RAGPipeline:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        #Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        #Initialize vector store
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )
        #Initialize LLM 
        self.llm = OllamaLLM(model="gemma3:latest")

        # Create RAG chain
        self.rag_chain = self.create_rag_chain()

    def create_rag_chain(self):
        """Create RAG chain"""
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Use the following pieces of retrieved context to answer the question at the end.
            You are an helpful assistant, so if you don't know the answer, just say that you don't know.
            Do not hallucinate. Do not make up information. Do not guess. Do not lie.
            Use factual information to answer the question. Verify the information you provide.
            Always cite the source of your answer in the format [Source: source_name]".
            
            Context: {context}

            Question: {question}

            Answer:
            """
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": self.vector_store.as_retriever(search_kwargs={"k": 4}) | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return self.rag_chain
    

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        self.vector_store.add_documents(documents)
        # In newer versions of langchain-chroma, persist() is no longer needed
        # as documents are automatically persisted when added

    
    def query(self, question: str) -> dict:
        """Query the RAG pipeline with a question"""
        # Get relevant documents
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        source_docs = retriever.invoke(question)
        
        # Get answer from chain
        answer = self.rag_chain.invoke(question)
        
        return {
            "answer": answer,
            "sources": source_docs
        }
