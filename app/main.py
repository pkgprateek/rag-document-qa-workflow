import gradio as gr
from rag_pipeline import RAGPipeline
from document_processor import DocumentProcessor
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Vertical configurations
VERTICALS = {
    "Legal": {
        "icon": "⚖️",
        "samples": [
            "data/samples/legal/service_agreement.txt",
            "data/samples/legal/amendment.txt",
            "data/samples/legal/nda.txt",
        ],
        "queries": [
            "What are the key termination conditions and notice periods?",
            "Summarize all payment terms, rates, and schedules",
        ],
    },
    "Research": {
        "icon": "🔬",
        "samples": [
            "data/samples/research/llm_enterprise_survey.txt",
            "data/samples/research/rag_methodology.txt",
            "data/samples/research/vector_db_benchmark.txt",
        ],
        "queries": [
            "What is the main research methodology used in these studies?",
            "Summarize the key findings and conclusions",
        ],
    },
    "FinOps": {
        "icon": "💰",
        "samples": [
            "data/samples/finops/cloud_cost_optimization.txt",
            "data/samples/finops/aws_invoice_sept2024.txt",
            "data/samples/finops/kubernetes_cost_allocation.txt",
        ],
        "queries": [
            "What are the top 3 cost optimization opportunities?",
            "Extract total spend by service category",
        ],
    },
}


class DocumentRagApp:
    def __init__(self):
        """
        Initialize Document RAG application with processor and pipeline.
        """
        self.processor = DocumentProcessor()
        self.rag_pipeline = RAGPipeline()
        self.loaded_documents = []
        self.current_vertical = "Legal"

    def load_sample_documents(self, vertical):
        """
        Load sample documents for a vertical.

        Args:
            vertical: Vertical name (Legal, Research, FinOps)

        Returns:
            str: Status message
        """
        try:
            samples = VERTICALS[vertical]["samples"]
            loaded_count = 0

            for sample_path in samples:
                if os.path.exists(sample_path):
                    chunks = self.processor.process_txt(sample_path)
                    self.rag_pipeline.add_documents(chunks, is_sample=True)
                    self.loaded_documents.append(os.path.basename(sample_path))
                    loaded_count += 1

            self.current_vertical = vertical
            icon = VERTICALS[vertical]["icon"]
            return f"{icon} Loaded {loaded_count} sample documents for **{vertical}** vertical"
        except Exception as e:
            return f"Error loading samples: {str(e)}"

    def process_document(self, file):
        """
        Process uploaded document (PDF/DOCX/TXT) and add to RAG system.

        Args:
            file: Gradio file upload object

        Returns:
            str: Status message with processing results or error
        """
        if file is None:
            return "Please upload a file."
        try:
            file_path = file.name
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()

            # Check file type and process the file based on its extension:
            if file_ext == ".pdf":
                chunks = self.processor.process_pdf(file_path)
            elif file_ext == ".txt":
                chunks = self.processor.process_txt(file_path)
            elif file_ext == ".docx":
                chunks = self.processor.process_docx(file_path)
            else:
                return "❌ Unsupported file type. Please upload PDF, TXT, or DOCX."

            self.rag_pipeline.add_documents(chunks, is_sample=False)
            self.loaded_documents.append(file_name)
            return f"✅ Processed **{len(chunks)} chunks** from `{file_name}`"
        except Exception as e:
            return f"❌ Error processing file: {str(e)}"

    def ask_question(self, question):
        """
        Answer user question using RAG pipeline with rate limiting.

        Args:
            question: User's question string

        Returns:
            str: Generated answer or error message
        """
        if not self.loaded_documents:
            return "⚠️ Please load sample documents or upload your own files first."

        if not question.strip():
            return "⚠️ Please enter a question."

        try:
            result = self.rag_pipeline.query(question)
            answer = result["answer"]
            return answer
        except Exception as e:
            return f"❌ Error answering question: {str(e)}"


# Initialize app
app = DocumentRagApp()

# Custom CSS for premium styling
custom_css = """
#hero-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

#hero-subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #6B7280;
    margin-bottom: 2rem;
}

.vertical-tab {
    font-size: 1.1rem;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    transition: all 0.2s;
}

.canned-query-btn {
    margin: 0.5rem;
    padding: 0.75rem 1rem;
    font-size: 0.95rem;
}

#how-it-works {
    background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%);
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
}

.step-item {
    display: inline-block;
    margin: 0 1.5rem;
    text-align: center;
}

.step-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}

#privacy-notice {
    background: #FEF3C7;
    border-left: 4px solid #F59E0B;
    padding: 1rem;
    border-radius: 6px;
    font-size: 0.9rem;
    margin-top: 1rem;
}

#calendly-badge {
    background: #3B82F6;
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
    margin-top: 1rem;
}

Footer {
    visibility: hidden;
}
"""

# Create Gradio Interface
with gr.Blocks(
    title="Enterprise RAG Platform", css=custom_css, theme=gr.themes.Soft()
) as demo:
    # Hero Section
    gr.Markdown("# Enterprise RAG + Agentic Automation", elem_id="hero-title")
    gr.Markdown(
        "Live demo for Legal | Research | FinOps teams — See intelligent document analysis in action",
        elem_id="hero-subtitle",
    )

    # Vertical Tabs
    with gr.Tabs() as tabs:
        with gr.Tab(f"{VERTICALS['Legal']['icon']} Legal", id="legal-tab"):
            load_legal_btn = gr.Button(
                "Load Legal Sample Documents", variant="primary", size="lg"
            )
            legal_status = gr.Markdown("")

        with gr.Tab(f"{VERTICALS['Research']['icon']} Research", id="research-tab"):
            load_research_btn = gr.Button(
                "Load Research Sample Documents", variant="primary", size="lg"
            )
            research_status = gr.Markdown("")

        with gr.Tab(f"{VERTICALS['FinOps']['icon']} FinOps", id="finops-tab"):
            load_finops_btn = gr.Button(
                "Load FinOps Sample Documents", variant="primary", size="lg"
            )
            finops_status = gr.Markdown("")

    gr.Markdown("---")

    # Main Demo Area
    with gr.Row():
        # Left Column: How It Works + Actions
        with gr.Column(scale=1):
            gr.Markdown("### 🌟 How It Works", elem_id="how-it-works")
            gr.Markdown("""
            <div style="text-align: center; padding: 1rem;">
                <div style="margin: 1rem 0;">
                    <span style="font-size: 2.5rem;">📄</span>
                    <p style="margin: 0.5rem 0; font-weight: 600;">1. Upload Documents</p>
                    <p style="font-size: 0.85rem; color: #6B7280;">PDF, DOCX, TXT files</p>
                </div>
                <div style="margin: 1rem 0; font-size: 2rem;">↓</div>
                <div style="margin: 1rem 0;">
                    <span style="font-size: 2.5rem;">🧠</span>
                    <p style="margin: 0.5rem 0; font-weight: 600;">2. AI Processes</p>
                    <p style="font-size: 0.85rem; color: #6B7280;">Chunks + Embeddings</p>
                </div>
                <div style="margin: 1rem 0; font-size: 2rem;">↓</div>
                <div style="margin: 1rem 0;">
                    <span style="font-size: 2.5rem;">💬</span>
                    <p style="margin: 0.5rem 0; font-weight: 600;">3. Ask Smart Questions</p>
                    <p style="font-size: 0.85rem; color: #6B7280;">Get cited answers in &lt;5 sec</p>
                </div>
            </div>
            """)

            gr.Markdown("### 📂 Or Upload Your Own")
            file_upload = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".docx", ".txt"],
                file_count="single",
            )
            process_btn = gr.Button("Process Document", variant="secondary")
            process_response = gr.Markdown("")

            # Calendly Badge
            gr.Markdown("""
            <div id="calendly-badge">
                <div style="text-align: center;">
                    📅 <strong>Paid Pilots Open</strong><br>
                    <a href="#" style="color: white; text-decoration: underline;" target="_blank">
                        Book 15-min Discovery Call →
                    </a>
                </div>
            </div>
            """)

            # Privacy Notice
            gr.Markdown("""
            <div id="privacy-notice">
                <strong>🔒 Data Privacy:</strong> Documents are processed into text chunks and stored temporarily.
                User uploads are auto-deleted after 7 days. Sample documents persist for demo purposes.
                No data used for model training.
            </div>
            """)

        # Right Column: Q&A Interface
        with gr.Column(scale=2):
            gr.Markdown("### 💡 Try Pre-Loaded Queries or Ask Your Own")

            # Canned Query Buttons
            with gr.Row():
                canned_btn_1 = gr.Button(
                    "🔍 What are the key termination conditions?",
                    elem_classes="canned-query-btn",
                )
                canned_btn_2 = gr.Button(
                    "💵 Summarize payment terms", elem_classes="canned-query-btn"
                )
            with gr.Row():
                canned_btn_3 = gr.Button(
                    "🔬 What methodology was used?", elem_classes="canned-query-btn"
                )
                canned_btn_4 = gr.Button(
                    "📊 Summarize key findings", elem_classes="canned-query-btn"
                )
            with gr.Row():
                canned_btn_5 = gr.Button(
                    "💰 Top 3 cost optimizations?", elem_classes="canned-query-btn"
                )
                canned_btn_6 = gr.Button(
                    "📈 Extract spend by category", elem_classes="canned-query-btn"
                )

            gr.Markdown("### ✍️ Custom Question")
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask anything about the loaded documents...",
                lines=3,
                scale=2,
            )
            ask_btn = gr.Button("Ask Question", variant="primary", size="lg")

            gr.Markdown("### 📜 Answer")
            answer_output = gr.Markdown("", container=True, min_height=400)

    # Event Handlers

    # Load sample documents
    load_legal_btn.click(
        fn=lambda: app.load_sample_documents("Legal"), outputs=[legal_status]
    )
    load_research_btn.click(
        fn=lambda: app.load_sample_documents("Research"), outputs=[research_status]
    )
    load_finops_btn.click(
        fn=lambda: app.load_sample_documents("FinOps"), outputs=[finops_status]
    )

    # Upload custom document
    process_btn.click(
        fn=app.process_document, inputs=[file_upload], outputs=[process_response]
    )

    # Canned queries
    canned_btn_1.click(
        fn=app.ask_question,
        inputs=[
            gr.Textbox(
                value="What are the key termination conditions and notice periods?",
                visible=False,
            )
        ],
        outputs=[answer_output],
    )
    canned_btn_2.click(
        fn=app.ask_question,
        inputs=[
            gr.Textbox(
                value="Summarize all payment terms, rates, and schedules", visible=False
            )
        ],
        outputs=[answer_output],
    )
    canned_btn_3.click(
        fn=app.ask_question,
        inputs=[
            gr.Textbox(
                value="What is the main research methodology used in these studies?",
                visible=False,
            )
        ],
        outputs=[answer_output],
    )
    canned_btn_4.click(
        fn=app.ask_question,
        inputs=[
            gr.Textbox(
                value="Summarize the key findings and conclusions", visible=False
            )
        ],
        outputs=[answer_output],
    )
    canned_btn_5.click(
        fn=app.ask_question,
        inputs=[
            gr.Textbox(
                value="What are the top 3 cost optimization opportunities?",
                visible=False,
            )
        ],
        outputs=[answer_output],
    )
    canned_btn_6.click(
        fn=app.ask_question,
        inputs=[
            gr.Textbox(value="Extract total spend by service category", visible=False)
        ],
        outputs=[answer_output],
    )

    # Custom question
    ask_btn.click(fn=app.ask_question, inputs=[question_input], outputs=[answer_output])

if __name__ == "__main__":
    demo.launch(share=False)
