import gradio as gr
from rag_pipeline import RAGPipeline
from document_processor import DocumentProcessor
import os
from dotenv import load_dotenv

load_dotenv()

# Vertical configurations
VERTICALS = {
    "Legal": [
        "data/samples/legal/service_agreement.txt",
        "data/samples/legal/amendment.txt",
        "data/samples/legal/nda.txt",
    ],
    "Research": [
        "data/samples/research/llm_enterprise_survey.txt",
        "data/samples/research/rag_methodology.txt",
        "data/samples/research/vector_db_benchmark.txt",
    ],
    "FinOps": [
        "data/samples/finops/cloud_cost_optimization.txt",
        "data/samples/finops/aws_invoice_sept2024.txt",
        "data/samples/finops/kubernetes_cost_allocation.txt",
    ],
}

QUERIES = {
    "Legal": ["What are the termination conditions?", "Summarize payment terms"],
    "Research": ["What methodology was used?", "Summarize key findings"],
    "FinOps": ["Top 3 cost optimizations?", "Extract spend by category"],
}


class DocumentRagApp:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.rag_pipeline = RAGPipeline()
        self.loaded_documents = []

    def load_samples(self, vertical):
        try:
            for path in VERTICALS[vertical]:
                if os.path.exists(path):
                    chunks = self.processor.process_txt(path)
                    self.rag_pipeline.add_documents(chunks, is_sample=True)
                    self.loaded_documents.append(os.path.basename(path))
            return f"✅ Loaded {len(VERTICALS[vertical])} {vertical} documents"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def process_file(self, file):
        if not file:
            return "Please upload a file"
        try:
            ext = os.path.splitext(file.name)[1].lower()
            if ext == ".pdf":
                chunks = self.processor.process_pdf(file.name)
            elif ext == ".txt":
                chunks = self.processor.process_txt(file.name)
            elif ext == ".docx":
                chunks = self.processor.process_docx(file.name)
            else:
                return "Unsupported format"

            self.rag_pipeline.add_documents(chunks, is_sample=False)
            return f"✅ Processed {len(chunks)} chunks"
        except Exception as e:
            return f"❌ {str(e)}"

    def ask(self, question):
        if not self.loaded_documents:
            return "Please load documents first"
        if not question.strip():
            return "Please enter a question"
        try:
            result = self.rag_pipeline.query(question)
            return result["answer"]
        except Exception as e:
            return f"Error: {str(e)}"


app = DocumentRagApp()

# Ultra-minimal CSS
css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

#hero {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
    background: linear-gradient(to right, #EFF6FF, #F0FDF4);
    border-radius: 12px;
    margin-bottom: 2rem;
}

#hero h1 {
    font-size: 2.25rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.5rem;
}

#hero p {
    font-size: 1.1rem;
    color: #6B7280;
}

.tab-nav button {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}

button {
    border-radius: 8px !important;
}

.primary-action {
    background: linear-gradient(to right, #2563EB, #059669) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border: none !important;
}

.query-btn {
    background: white !important;
    border: 2px solid #E5E7EB !important;
    color: #374151 !important;
    text-align: left !important;
    padding: 0.65rem 1rem !important;
    font-size: 0.95rem !important;
}

.query-btn:hover {
    border-color: #2563EB !important;
    background: #F9FAFB !important;
}

#answer-area {
    background: white;
    border: 2px solid #E5E7EB;
    border-radius: 10px;
    padding: 1.5rem;
    min-height: 350px;
    line-height: 1.7;
}

#info-box {
    background: #FFFBEB;
    border-left: 4px solid #F59E0B;
    padding: 1rem;
    border-radius: 6px;
    margin-top: 1rem;
    font-size: 0.9rem;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(), title="Enterprise RAG Demo") as demo:
    # Hero
    gr.HTML("""
        <div id="hero">
            <h1>Enterprise RAG + Agentic Automation</h1>
            <p>Document intelligence for Legal, Research, and FinOps teams</p>
        </div>
    """)

    # Tabs
    with gr.Tabs():
        for vertical in ["Legal", "Research", "FinOps"]:
            icon = {"Legal": "⚖️", "Research": "🔬", "FinOps": "💰"}[vertical]
            with gr.Tab(f"{icon} {vertical}"):
                gr.Button(
                    f"Load {vertical} Samples", elem_classes="primary-action", size="lg"
                ).click(
                    fn=lambda v=vertical: app.load_samples(v), outputs=gr.Markdown("")
                )

    gr.Markdown("---")

    # Main area
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Quick Queries")

            # 6 query buttons (2 rows of 3)
            with gr.Row():
                q1 = gr.Button(
                    "What are the termination conditions?", elem_classes="query-btn"
                )
                q2 = gr.Button("Summarize payment terms", elem_classes="query-btn")
                q3 = gr.Button("What methodology was used?", elem_classes="query-btn")
            with gr.Row():
                q4 = gr.Button("Summarize key findings", elem_classes="query-btn")
                q5 = gr.Button("Top 3 cost optimizations?", elem_classes="query-btn")
                q6 = gr.Button("Extract spend by category", elem_classes="query-btn")

            gr.Markdown("### ✍️ Custom Question")
            question = gr.Textbox(
                placeholder="Ask anything about loaded documents...",
                show_label=False,
                lines=2,
            )
            gr.Button("Ask", elem_classes="primary-action").click(
                fn=app.ask,
                inputs=question,
                outputs=gr.Markdown("", elem_id="answer-area"),
            )

            gr.Markdown("### 📜 Answer", elem_id="answer-header")
            answer = gr.Markdown(
                "*Load documents above to start*", elem_id="answer-area"
            )

        with gr.Column(scale=1):
            gr.Markdown("### 📂 Upload")
            file = gr.File(file_types=[".pdf", ".docx", ".txt"])
            gr.Button("Process", elem_classes="primary-action").click(
                fn=app.process_file, inputs=file, outputs=gr.Markdown("")
            )

            gr.HTML("""
                <div style="background: linear-gradient(135deg, #2563EB, #059669); color: white; padding: 1.25rem; border-radius: 10px; text-align: center; margin-top: 1.5rem;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📅</div>
                    <div style="font-weight: 700; margin-bottom: 0.5rem;">Paid Pilots Open</div>
                    <a href="#" style="color: white; text-decoration: underline;">Book 15-min Call →</a>
                </div>
            """)

            gr.HTML("""
                <div id="info-box">
                    <strong>🔒 Privacy:</strong> Documents processed into text chunks, auto-deleted after 7 days. No data used for training.
                </div>
            """)

    # Wire up queries
    for i, btn in enumerate([q1, q2, q3, q4, q5, q6]):
        queries_list = QUERIES["Legal"] + QUERIES["Research"] + QUERIES["FinOps"]
        btn.click(fn=lambda q=queries_list[i]: app.ask(q), outputs=answer)

if __name__ == "__main__":
    demo.launch(share=False)
