from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from typing import List
import os
from datetime import datetime, timedelta
import json
from pathlib import Path

# Fix tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RAGPipeline:
    # Model configuration for multi-provider support
    MODEL_CONFIG = {
        "gpt-oss-120b": {
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "display": "GPT-OSS 120B (OpenAI)",
            "temperature": 0.1,
            "max_tokens": 1024,
        },
        "llama-3.3-70b": {
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "display": "Llama 3.3 70B (Meta)",
            "temperature": 0.1,
            "max_tokens": 1024,
        },
        "gemma-3-27b": {
            "provider": "openrouter",
            "model": "google/gemma-3-27b-it:free",
            "display": "Gemma 3 27B (Google)",
            "temperature": 0.1,
            "max_tokens": 512,
        },
    }

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        default_model: str = "gpt-oss-120b",
    ):
        """
        Initialize RAG pipeline with embeddings, vector store, and multi-provider LLM support.
        Sets up rate limiting (10 queries/hour) and supports Groq + OpenRouter APIs.

        Args:
            persist_directory: Path to store ChromaDB vector database (default: ./data/chroma_db)
            default_model: Model key from MODEL_CONFIG (default: gpt-oss-120b)
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

        # Initialize LLM with default model
        self.current_model = default_model
        self.llm = self._initialize_llm(default_model)

        # Current session ID for retrieval filtering (set per-query)
        self._current_session_id = None

        # Create RAG chain
        self.rag_chain = self.create_rag_chain()

    def _initialize_llm(self, model_key: str):
        """
        Initialize LLM based on provider and model configuration.
        Supports both Groq and OpenRouter providers.

        Args:
            model_key: Key from MODEL_CONFIG dictionary

        Returns:
            ChatOpenAI: Configured LLM instance

        Raises:
            ValueError: If model_key is invalid or required API key is missing
        """
        if model_key not in self.MODEL_CONFIG:
            raise ValueError(
                f"Invalid model key: {model_key}. "
                f"Available models: {', '.join(self.MODEL_CONFIG.keys())}"
            )

        config = self.MODEL_CONFIG[model_key]
        provider = config["provider"]

        if provider == "groq":
            # Groq API configuration
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                raise ValueError(
                    "GROQ_API_KEY environment variable not set. "
                    "Get one free at https://console.groq.com/keys"
                )

            return ChatOpenAI(
                model=config["model"],
                openai_api_key=groq_key,
                openai_api_base="https://api.groq.com/openai/v1",
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            )

        elif provider == "openrouter":
            # OpenRouter API configuration
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_key:
                raise ValueError(
                    "OPENROUTER_API_KEY environment variable not set. "
                    "Get one free at https://openrouter.ai/keys"
                )

            return ChatOpenAI(
                model=config["model"],
                openai_api_key=openrouter_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def switch_model(self, model_key: str) -> str:
        """
        Dynamically switch to a different LLM model and recreate the RAG chain.

        Args:
            model_key: Key from MODEL_CONFIG dictionary

        Returns:
            str: Display name of the switched model

        Raises:
            ValueError: If model_key is invalid or API key is missing
        """
        # Initialize new LLM
        self.llm = self._initialize_llm(model_key)
        self.current_model = model_key

        # Recreate RAG chain with new LLM
        self.rag_chain = self.create_rag_chain()

        return self.MODEL_CONFIG[model_key]["display"]

    def create_rag_chain(self):
        """
        Creates the RAG chain by combining retriever, prompt template, and LLM.

        Returns:
            RunnableParallel: Chain that retrieves context and generates answers
        """
        prompt = PromptTemplate(
            input_variables=["context", "sources", "question"],
            template="""You are an expert AI assistant specializing in document analysis. Your goal is to provide comprehensive, accurate, and well-cited answers.

Available Documents: {sources}

Context from Documents:
{context}

User Question: {question}

INSTRUCTIONS FOR YOUR RESPONSE:
1. **Analyze Thoroughly**: Read the context carefully and identify all relevant information
2. **Answer Comprehensively**: Provide a complete, detailed answer that fully addresses the question
3. **Use Proper Structure**: 
   - Start with a clear, direct answer
   - Follow with supporting details and explanation
   - Use markdown formatting (headings, bullet points, bold) for readability
4. **Cite Sources Inline**: As you make specific claims, cite the source immediately
   - Format: (Source: filename, Page X) or (Source: filename) if page unknown
   - Example: "The termination period is 30 days (Source: service_agreement.pdf, Page 3)"
   - Be specific about which document and page number whenever possible
5. **Include a Sources Section**: At the end of your answer, add:
   **Sources Referenced:**
   • filename (Page X) - Brief note about what info came from here
   • filename2 (Page Y) - Brief note
   
6. **Quality Standards**:
   - Be specific and precise with facts, numbers, dates, and terms
   - Quote exact phrases when important (use quotation marks)
   - If information is unclear or missing, state what's uncertain
   - Connect related points to create a cohesive narrative

Answer:""",
        )

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
        )

        # Wrap retriever to filter by session
        def session_filter(docs):
            """Filter documents by current session."""
            session_id = self._current_session_id
            if session_id:
                # Return docs matching session_id OR sample docs (is_sample=True)
                return [
                    d
                    for d in docs
                    if d.metadata.get("session_id") == session_id
                    or d.metadata.get("is_sample", False)
                ]
            return docs

        # Create session-filtered retriever as a Runnable
        session_filtered_retriever = retriever | RunnableLambda(session_filter)

        rag_chain = RunnableParallel(
            {
                "result": (
                    {
                        "context": session_filtered_retriever
                        | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                        "sources": session_filtered_retriever
                        | (
                            lambda docs: ", ".join(
                                list(
                                    set(
                                        [
                                            d.metadata.get("source", "").split("/")[-1]
                                            for d in docs
                                        ]
                                    )
                                )
                            )
                        ),
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | self.llm
                ),
                "source_documents": session_filtered_retriever,
            }
        )
        return rag_chain

    def add_documents(
        self,
        documents: List[Document],
        session_id: str = None,
        is_sample: bool = False,
    ) -> None:
        """
        Add processed document chunks to the vector store for retrieval.
        Adds session_id and timestamp metadata for isolation and auto-cleanup.

        Args:
            documents: List of Document objects with text and metadata
            session_id: User's session ID for isolation (None for samples)
            is_sample: If True, document is global and won't be auto-deleted
        """
        # Add session and timestamp metadata to each chunk
        now = datetime.now().isoformat()

        for doc in documents:
            doc.metadata["session_id"] = session_id if not is_sample else "global"
            doc.metadata["uploaded_at"] = now
            doc.metadata["is_sample"] = is_sample

        self.vector_store.add_documents(documents)

        # Track document metadata for cleanup (skip samples)
        if not is_sample and documents:
            self._track_document(
                documents[0].metadata.get("source", "unknown"),
                session_id=session_id,
            )

    def _check_rate_limit(self) -> bool:
        """
        Enforces rate limit of 10 queries per hour by tracking query timestamps.

        Returns:
            bool: True if within limit, False if exceeded
        """
        now = datetime.now()

        # Load existing queries if file exists
        if self.rate_limit_file.exists():
            try:
                with open(self.rate_limit_file, "r") as f:
                    content = f.read().strip()
                    if content:  # Only parse if file is not empty
                        data = json.loads(content)
                        queries = [
                            datetime.fromisoformat(q) for q in data.get("queries", [])
                        ]
                    else:
                        queries = []
            except (json.JSONDecodeError, ValueError):
                # If file is corrupted, start fresh
                queries = []
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

    def query(self, question: str, session_id: str = None):
        """
        Query the RAG system with a question, retrieves relevant context and generates answer.
        Results are filtered to the user's session documents + global samples.

        Args:
            question: User's question string
            session_id: User's session ID for filtering results

        Returns:
            dict: {
                "answer": str,
                "citations": List[dict],
                "num_sources": int
            }

        Raises:
            ValueError: If rate limit (10 queries/hour) is exceeded
        """
        # Check rate limit
        if not self._check_rate_limit():
            raise ValueError(
                "Rate limit exceeded. You can only ask 10 questions per hour. "
                "Please try again later."
            )

        # Set session ID for filtered retrieval
        self._current_session_id = session_id

        answer = self.rag_chain.invoke(question)
        result = answer["result"]

        # Extract answer text
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

    def query_stream(self, question: str, session_id: str = None):
        """
        Stream answer tokens for real-time display.
        Yields tokens as they arrive from the LLM.

        Args:
            question: User's question string
            session_id: User's session ID for filtering results

        Yields:
            str: Accumulated answer text (each yield contains full answer so far)
        """
        # Check rate limit
        if not self._check_rate_limit():
            yield "⚠️ Rate limit exceeded. You can only ask 10 questions per hour. Please try again later."
            return

        # Set session ID for filtered retrieval
        self._current_session_id = session_id

        # Get documents using retriever (non-streaming part)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)

        # Filter by session
        if session_id:
            docs = [
                d
                for d in docs
                if d.metadata.get("session_id") == session_id
                or d.metadata.get("is_sample", False)
            ]

        if not docs:
            yield "I couldn't find relevant information in your documents. Please try rephrasing your question."
            return

        # Build context and sources
        context = "\n\n".join([d.page_content for d in docs])
        sources = ", ".join(
            list(set([d.metadata.get("source", "").split("/")[-1] for d in docs]))
        )

        # Format prompt
        prompt = self._format_prompt(context, sources, question)

        # Stream from LLM
        full_answer = ""
        for chunk in self.llm.stream(prompt):
            if hasattr(chunk, "content"):
                full_answer += chunk.content
            else:
                full_answer += str(chunk)
            yield full_answer

    def _format_prompt(self, context: str, sources: str, question: str) -> str:
        """
        Format the RAG prompt with context, sources, and question.

        Args:
            context: Retrieved document content
            sources: Comma-separated source filenames
            question: User's question

        Returns:
            str: Formatted prompt string
        """
        return f"""You are an expert AI assistant specializing in document analysis. Your goal is to provide comprehensive, accurate, and well-cited answers.

Available Documents: {sources}

Context from Documents:
{context}

User Question: {question}

INSTRUCTIONS FOR YOUR RESPONSE:
1. **Analyze Thoroughly**: Read the context carefully and identify all relevant information
2. **Answer Comprehensively**: Provide a complete, detailed answer that fully addresses the question
3. **Use Proper Structure**: 
   - Start with a clear, direct answer
   - Follow with supporting details and explanation
   - Use markdown formatting (headings, bullet points, bold) for readability
4. **Cite Sources Inline**: As you make specific claims, cite the source immediately
   - Format: (Source: filename, Page X) or (Source: filename) if page unknown
   - Example: "The termination period is 30 days (Source: service_agreement.pdf, Page 3)"
   - Be specific about which document and page number whenever possible
5. **Include a Sources Section**: At the end of your answer, add:
   **Sources Referenced:**
   • filename (Page X) - Brief note about what info came from here
   • filename2 (Page Y) - Brief note
   
6. **Quality Standards**:
   - Be specific and precise with facts, numbers, dates, and terms
   - Quote exact phrases when important (use quotation marks)
   - If information is unclear or missing, state what's uncertain
   - Connect related points to create a cohesive narrative

Answer:"""

    def _extract_citations(self, source_documents: List[Document]) -> List[dict]:
        """
        Extract formatted citations from source documents with page numbers and previews.

        Args:
            source_documents: List of retrieved Document objects from RAG chain

        Returns:
            List[dict]: Formatted citations with id, source, page, and preview
        """
        import re

        citations = []

        for idx, doc in enumerate(source_documents, 1):
            # Extract file name (basename only)
            source_path = doc.metadata.get("source", "Unknown")
            file_name = (
                source_path.split("/")[-1] if "/" in source_path else source_path
            )

            # Parse page number from content (PDF format: "---- Page X ----")
            page_num = None
            content = doc.page_content

            # Try direct metadata first
            if "page" in doc.metadata:
                page_num = str(doc.metadata["page"])
            # Fallback: parse from content markers
            elif "---- Page " in content:
                match = re.search(r"---- Page (\d+) ----", content)
                if match:
                    page_num = match.group(1)

            # Get clean preview (remove page markers)
            preview = re.sub(r"---- Page \d+ ----", "", content).strip()
            # Take first 150 chars for preview
            if len(preview) > 150:
                preview = preview[:150] + "..."

            citations.append(
                {
                    "id": idx,
                    "source": file_name,
                    "page": page_num,
                    "preview": preview,
                    "full_content": content,
                }
            )

        return citations

    def _track_document(self, source_path: str, session_id: str = None) -> None:
        """
        Track document upload timestamp for auto-cleanup.

        Args:
            source_path: Path to the uploaded document
            session_id: User's session ID for the document
        """
        # Load existing metadata
        if self.doc_metadata_file.exists():
            with open(self.doc_metadata_file, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {"documents": {}}

        # Add new document with current timestamp and session
        metadata["documents"][source_path] = {
            "uploaded_at": datetime.now().isoformat(),
            "session_id": session_id,
            "is_sample": False,
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
        deleted_count = 0

        for doc_path, doc_info in metadata.get("documents", {}).items():
            upload_time = datetime.fromisoformat(doc_info["uploaded_at"])

            # Keep if uploaded within 7 days OR is a sample
            if upload_time > seven_days_ago or doc_info.get("is_sample", False):
                documents_to_keep[doc_path] = doc_info
            else:
                # Actually delete from ChromaDB using source path filter
                try:
                    self.vector_store._collection.delete(where={"source": doc_path})
                    deleted_count += 1
                    print(f"Deleted expired document: {doc_path}")
                except Exception as e:
                    print(f"Error deleting document {doc_path}: {e}")

        # Update metadata file
        metadata["documents"] = documents_to_keep
        with open(self.doc_metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        if deleted_count > 0:
            print(f"Cleanup complete: removed {deleted_count} expired documents")

    def get_documents_by_session(self, session_id: str) -> List[str]:
        """
        Get list of document names for a given session.

        Args:
            session_id: User's session ID

        Returns:
            List[str]: List of document filenames belonging to this session
        """
        if not self.doc_metadata_file.exists():
            return []

        with open(self.doc_metadata_file, "r") as f:
            metadata = json.load(f)

        documents = []
        for doc_path, doc_info in metadata.get("documents", {}).items():
            if doc_info.get("session_id") == session_id:
                # Extract just the filename
                filename = doc_path.split("/")[-1] if "/" in doc_path else doc_path
                documents.append(
                    {
                        "filename": filename,
                        "path": doc_path,
                        "uploaded_at": doc_info["uploaded_at"],
                    }
                )

        return documents

    def delete_document(self, session_id: str, source_path: str) -> bool:
        """
        Delete a specific document from vector store and metadata.

        Args:
            session_id: User's session ID (for verification)
            source_path: Full path to the document to delete

        Returns:
            bool: True if deleted, False if not found or not authorized
        """
        if not self.doc_metadata_file.exists():
            return False

        with open(self.doc_metadata_file, "r") as f:
            metadata = json.load(f)

        # Verify document belongs to this session
        doc_info = metadata.get("documents", {}).get(source_path)
        if not doc_info:
            return False
        if doc_info.get("session_id") != session_id:
            return False  # Not authorized to delete

        # Delete from ChromaDB
        try:
            self.vector_store._collection.delete(where={"source": source_path})
        except Exception as e:
            print(f"Error deleting from ChromaDB: {e}")
            return False

        # Remove from metadata
        del metadata["documents"][source_path]
        with open(self.doc_metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return True
