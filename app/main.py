import gradio as gr
from rag_pipeline import RAGPipeline
from document_processor import DocumentProcessor
import os
from dotenv import load_dotenv

load_dotenv()


class DocumentRagApp:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.rag_pipeline = RAGPipeline()
        self.loaded_documents = []

    def load_samples(self, vertical):
        samples = {
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

        try:
            for path in samples[vertical]:
                if os.path.exists(path):
                    chunks = self.processor.process_txt(path)
                    self.rag_pipeline.add_documents(chunks, is_sample=True)
                    self.loaded_documents.append(os.path.basename(path))
            return f"✓ Loaded {len(samples[vertical])} {vertical} documents"
        except Exception as e:
            return f"Error: {str(e)}"

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
            return f"✓ Processed {len(chunks)} chunks"
        except Exception as e:
            return f"Error: {str(e)}"

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

# ChatGPT-inspired dark theme
css = """
:root {
    --bg-dark: #343541;
    --bg-darker: #202123;
    --bg-input: #40414F;
    --text: #ECECF1;
    --text-dim: #A0A0AA;
    --border: #565869;
    --accent: #19C37D;
}

.gradio-container {
    background: var(--bg-dark) !important;
    font-family: -apple-system, system-ui, sans-serif !important;
    max-width: 100% !important;
    padding: 0 !important;
}

#main-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
}

/* Header */
#header {
    text-align: center;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}

#header h1 {
    color: var(--text);
    font-size: 1.75rem;
    font-weight: 600;
    margin: 0 0 0.5rem 0;
}

#header p {
    color: var(--text-dim);
    font-size: 0.95rem;
    margin: 0;
}

/* Controls section */
.controls {
    background: var(--bg-input);
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border);
}

.controls-title {
    color: var(--text);
    font-size: 0.875rem;
    font-weight: 600;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Dropdown and buttons */
select, button, textarea, input {
    background: var(--bg-darker) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

select:focus, textarea:focus, input:focus {
    border-color: var(--accent) !important;
    outline: none !important;
}

button {
    padding: 0.625rem 1.25rem !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}

button:hover {
    background: var(--bg-input) !important;
    border-color: var(--accent) !important;
}

.primary-btn {
    background: var(--accent) !important;
    color: #000 !important;
    font-weight: 600 !important;
}

.primary-btn:hover {
    background: #1AB370 !important;
}

/* Query buttons */
.query-btn {
    width: 100% !important;
    text-align: left !important;
    margin-bottom: 0.5rem !important;
}

/* Question input */
#question-box {
    background: var(--bg-input);
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border);
}

textarea {
    font-size: 1rem !important;
    line-height: 1.5 !important;
    padding: 0.75rem !important;
}

/* Answer area */
#answer-section {
    background: var(--bg-input);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    border: 1px solid var(--border);
    min-height: 300px;
}

#answer-section .markdown {
    color: var(--text) !important;
    line-height: 1.7;
    font-size: 0.95rem;
}

/* Footer info */
#footer-info {
    max-width: 800px;
    margin: 2rem auto 0;
    padding: 2rem 1.5rem;
    border-top: 1px solid var(--border);
}

.info-box {
    background: var(--bg-input);
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
    font-size: 0.875rem;
    color: var(--text-dim);
    line-height: 1.6;
}

.calendly-box {
    background: linear-gradient(135deg, #1A7F64, var(--accent));
    color: #000;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
    font-weight: 600;
}

.calendly-box a {
    color: #000;
    text-decoration: underline;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Base(), title="Enterprise RAG") as demo:
    with gr.Column(elem_id="main-container"):
        # Header
        gr.HTML("""
            <div id="header">
                <h1>Enterprise RAG Platform</h1>
                <p>Document intelligence for Legal, Research, and FinOps</p>
            </div>
        """)

        # Load samples
        with gr.Group(elem_classes="controls"):
            gr.HTML('<div class="controls-title">Load Sample Documents</div>')
            with gr.Row():
                sample_dropdown = gr.Dropdown(
                    choices=["Legal", "Research", "FinOps"],
                    value="Legal",
                    show_label=False,
                    scale=3,
                )
                load_btn = gr.Button("Load", elem_classes="primary-btn", scale=1)
            load_status = gr.Markdown("")

        # Upload
        with gr.Group(elem_classes="controls"):
            gr.HTML('<div class="controls-title">Or Upload Your Documents</div>')
            file_upload = gr.File(
                file_types=[".pdf", ".docx", ".txt"], show_label=False
            )
            process_btn = gr.Button("Process", elem_classes="primary-btn")
            upload_status = gr.Markdown("")

        # Quick queries
        with gr.Group(elem_classes="controls"):
            gr.HTML('<div class="controls-title">Quick Queries</div>')
            q1 = gr.Button(
                "What are the termination conditions?", elem_classes="query-btn"
            )
            q2 = gr.Button("Summarize payment terms", elem_classes="query-btn")
            q3 = gr.Button("What methodology was used?", elem_classes="query-btn")
            q4 = gr.Button("Summarize key findings", elem_classes="query-btn")
            q5 = gr.Button("Top 3 cost optimizations?", elem_classes="query-btn")
            q6 = gr.Button("Extract spend by category", elem_classes="query-btn")

        # Question
        with gr.Group(elem_id="question-box"):
            gr.HTML('<div class="controls-title">Ask Your Question</div>')
            question = gr.Textbox(
                placeholder="Type your question here...", show_label=False, lines=2
            )
            ask_btn = gr.Button("Ask", elem_classes="primary-btn")

        # Answer
        with gr.Group(elem_id="answer-section"):
            gr.HTML('<div class="controls-title">Answer</div>')
            answer = gr.Markdown("*Load documents to get started*")

    # Footer
    with gr.Column(elem_id="footer-info"):
        gr.HTML("""
            <div class="calendly-box">
                📅 2-Week Paid Pilots Available · 
                <a href="#" target="_blank">Book Discovery Call</a>
            </div>
        """)
        gr.HTML("""
            <div class="info-box">
                🔒 Privacy: Documents processed locally, auto-deleted after 7 days, never used for training
            </div>
        """)

    # Event handlers
    load_btn.click(fn=app.load_samples, inputs=sample_dropdown, outputs=load_status)
    process_btn.click(fn=app.process_file, inputs=file_upload, outputs=upload_status)

    q1.click(fn=lambda: app.ask("What are the termination conditions?"), outputs=answer)
    q2.click(fn=lambda: app.ask("Summarize payment terms"), outputs=answer)
    q3.click(fn=lambda: app.ask("What methodology was used?"), outputs=answer)
    q4.click(fn=lambda: app.ask("Summarize key findings"), outputs=answer)
    q5.click(fn=lambda: app.ask("Top 3 cost optimizations?"), outputs=answer)
    q6.click(fn=lambda: app.ask("Extract spend by category"), outputs=answer)

    ask_btn.click(fn=app.ask, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch(share=False)
