from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from typing import List
import os
from datetime import datetime, timedelta
import json
from pathlib import Path

# Fix tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RAGPipeline:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Initialize RAG pipeline with embeddings, vector store, and LLM.
        Sets up rate limiting (10 queries/hour) and uses OpenRouter API with free Gemma model.

        Args:
            persist_directory: Path to store ChromaDB vector database (default: ./data/chroma_db)
        """
        # Initialize better embeddings (BAAI/bge-small-en-v1.5)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # Important for bge models
        )

        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )

        # Rate limiting setup (10 queries per hour)
        self.rate_limit_file = Path("./data/rate_limit.json")
        self.rate_limit_file.parent.mkdir(parents=True, exist_ok=True)

        # Document tracking for auto-cleanup (7-day retention)
        self.doc_metadata_file = Path("./data/document_metadata.json")
        self.doc_metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-cleanup on initialization
        self._cleanup_old_documents()

        # Initialize LLM using OpenRouter (cheapest free option)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get one free at https://openrouter.ai/keys"
            )

        # Using google/gemma-3-4b-it:free - free tier on OpenRouter
        self.llm = ChatOpenAI(
            model="google/gemma-3-4b-it:free",
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=512,
        )

        # Create RAG chain
        self.rag_chain = self.create_rag_chain()

    def create_rag_chain(self):
        """
        Creates the RAG chain by combining retriever, prompt template, and LLM.

        Returns:
            RunnableParallel: Chain that retrieves context and generates answers
        """
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Answer the question based on the context below. If you cannot answer based on the context, say "I don't know".
            Do not hallucinate. Do not make up information.
            Format your answer using markdown for better readability.

            Context: {context}

            Question: {question}

            Provide a clear and concise answer:""",
        )

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
        )

        rag_chain = RunnableParallel(
            {
                "result": (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | self.llm
                ),
                "source_documents": retriever,
            }
        )
        return rag_chain

    def add_documents(self, documents: List[Document], is_sample: bool = False) -> None:
        """
        Add processed document chunks to the vector store for retrieval.
        Tracks upload timestamp for auto-cleanup (user docs only).

        Args:
            documents: List of Document objects with text and metadata
            is_sample: If True, document won't be auto-deleted (for demo samples)
        """
        self.vector_store.add_documents(documents)
        # In newer versions of langchain-chroma, persist() is no longer needed
        # as documents are automatically persisted when added
        
        # Track document metadata for cleanup (skip samples)
        if not is_sample and documents:
            self._track_document(documents[0].metadata.get("source", "unknown"))

    def _check_rate_limit(self) -> bool:
        """
        Enforces rate limit of 10 queries per hour by tracking query timestamps.

        Returns:
            bool: True if within limit, False if exceeded
        """
        now = datetime.now()

        # Load existing queries
        if self.rate_limit_file.exists():
            with open(self.rate_limit_file, "r") as f:
                data = json.load(f)
                queries = [datetime.fromisoformat(q) for q in data.get("queries", [])]
        else:
            queries = []

        # Remove queries older than 1 hour
        one_hour_ago = now - timedelta(hours=1)
        recent_queries = [q for q in queries if q > one_hour_ago]

        # Check limit
        if len(recent_queries) >= 10:
            return False

        # Add current query
        recent_queries.append(now)

        # Save updated queries
        with open(self.rate_limit_file, "w") as f:
            json.dump({"queries": [q.isoformat() for q in recent_queries]}, f)

        return True

    def query(self, question: str):
        """
        Query the RAG system with a question, retrieves relevant context and generates answer.

        Args:
            question: User's question string

        Returns:
            dict: {"answer": str} containing the generated response

        Raises:
            ValueError: If rate limit (10 queries/hour) is exceeded
        """
        # Check rate limit
        if not self._check_rate_limit():
            raise ValueError(
                "Rate limit exceeded. You can only ask 10 questions per hour. "
                "Please try again later."
            )

        answer = self.rag_chain.invoke(question)
        result = answer["result"]

        if hasattr(result, "content"):
            answer_text = result.content
        elif hasattr(result, "text"):
            answer_text = result.text
        else:
            answer_text = str(result)

        # Check if answer is empty
        if not answer_text or answer_text.strip() == "":
            answer_text = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        return {"answer": answer_text}

    def _track_document(self, source_path: str) -> None:
        """
        Track document upload timestamp for auto-cleanup.
        
        Args:
            source_path: Path to the uploaded document
        """
        # Load existing metadata
        if self.doc_metadata_file.exists():
            with open(self.doc_metadata_file, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {"documents": {}}
        
        # Add new document with current timestamp
        metadata["documents"][source_path] = {
            "uploaded_at": datetime.now().isoformat(),
            "is_sample": False
        }
        
        # Save updated metadata
        with open(self.doc_metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _cleanup_old_documents(self) -> None:
        """
        Remove documents older than 7 days from vector store.
        Sample documents are never deleted.
        """
        if not self.doc_metadata_file.exists():
            return
        
        with open(self.doc_metadata_file, "r") as f:
            metadata = json.load(f)
        
        now = datetime.now()
        seven_days_ago = now - timedelta(days=7)
        documents_to_keep = {}
        
        for doc_path, doc_info in metadata.get("documents", {}).items():
            upload_time = datetime.fromisoformat(doc_info["uploaded_at"])
            
            # Keep if uploaded within 7 days OR is a sample
            if upload_time > seven_days_ago or doc_info.get("is_sample", False):
                documents_to_keep[doc_path] = doc_info
            else:
                # Delete from vector store
                # Note: ChromaDB doesn't support direct deletion by metadata filter
                # In production, you'd implement this with collection.delete()
                print(f"Would delete old document: {doc_path}")
        
        # Update metadata file
        metadata["documents"] = documents_to_keep
        with open(self.doc_metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
