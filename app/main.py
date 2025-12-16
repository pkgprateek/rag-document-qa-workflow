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

# Premium Enterprise Design System (Restored & Cleaned)
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@400;500;600;700&display=swap');

:root {
    /* Brand Palette */
    --primary-gradient: linear-gradient(135deg, #3B82F6 0%, #10B981 100%);
    --surface-dark: #0B0F19;
    --surface-glass: rgba(17, 24, 39, 0.7);
    --border-glass: rgba(255, 255, 255, 0.08); 
    --text-primary: #F9FAFB;
    --text-secondary: #9CA3AF;
    --accent: #10B981;
    
    --font-heading: 'Outfit', sans-serif;
    --font-body: 'Inter', sans-serif;
}

/* --- GLOBAL RESET & CLEANING --- */
body, .gradio-container {
    background-color: var(--surface-dark) !important;
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}

/* ⚠️ CRITICAL: Remove Gradio's default nested boxes/backgrounds ⚠️ */
.gradio-container .block, 
.gradio-container .form, 
.gradio-container .gradio-box,
.gradio-container .padded,
.gradio-container .gradio-group,
.gradio-container .gradio-row,
.gradio-container .gradio-column {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Force transparency on specific internal elements to fix "Grey Box" issue */
.glass-card div {
    background-color: transparent !important;
    border: none !important;
}

/* Re-assert styles for Inputs/Buttons since the rule above is aggressive */
.glass-card textarea, 
.glass-card input[type="text"], 
.glass-card .gradio-dropdown {
    background-color: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid var(--border-glass) !important;
}

.glass-card button.primary-btn {
    background: var(--primary-gradient) !important;
}

.glass-card button.query-btn {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--border-glass) !important;
}


/* Typography */
h1, h2, h3, h4 { font-family: var(--font-heading); }
span, p, div { font-family: var(--font-body); }

/* --- HERO SECTION --- */
#header {
    text-align: center;
    margin-bottom: 4rem;
    padding-top: 2rem;
}
#header h1 {
    font-size: 3.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: #FFFFFF; /* High contrast white */
    text-shadow: 0 0 20px rgba(59, 130, 246, 0.5); /* Glow effect instead of gradient text */
    letter-spacing: -0.02em;
}
#header p {
    font-size: 1.2rem;
    color: var(--text-secondary);
}

/* --- GLASS CARDS --- */
.glass-card {
    background: var(--surface-glass) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass) !important;
    border-radius: 20px !important;
    padding: 2rem !important; /* Internal padding for the card content */
    margin-bottom: 2rem !important;
    box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5) !important;
    height: 100% !important; /* Attempt to stretch */
    display: flex !important;
    flex-direction: column !important;
}

.card-header {
    font-family: var(--font-heading);
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-secondary);
    margin-bottom: 0.5rem; /* Reduced bottom margin */
    border-bottom: 1px solid var(--border-glass);
    padding-bottom: 0.5rem;
}

/* --- INPUTS & BUTTONS (Cleaned) --- */
.gradio-dropdown, .gradio-textbox textarea {
    background-color: rgba(0, 0, 0, 0.3) !important; 
    border: 1px solid var(--border-glass) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* Upload Area specific */
.gradio-file {
    background-color: rgba(0, 0, 0, 0.2) !important;
    border: 2px dashed rgba(255, 255, 255, 0.3) !important; /* Brighter border */
    border-radius: 12px !important;
}

.gradio-dropdown:hover, .gradio-textbox textarea:hover {
    border-color: var(--accent) !important;
}

/* Primary Button */
.primary-btn {
    background: var(--primary-gradient) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 1rem !important;
    border-radius: 10px !important;
    transition: transform 0.2s;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
}
.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
}

/* Quick Query Buttons */
.query-btn {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--border-glass) !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    padding: 0.8rem !important;
    text-align: left !important;
    font-size: 0.95rem !important;
}
.query-btn:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    color: var(--text-primary) !important;
    border-color: var(--accent) !important;
}

/* --- TABS (Seamless) --- */
.tab-nav {
    border: none !important;
    margin-top: 1rem !important;    /* Spacing above tabs */
    margin-bottom: 1.5rem !important; /* Spacing below tabs */
    background: rgba(0,0,0,0.2) !important;
    border-radius: 12px;
    padding: 4px !important;
    display: flex;
    gap: 4px;
}
.tab-nav button {
    border: none !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border-radius: 8px !important;
    flex-grow: 1;
    font-family: var(--font-heading) !important;
}
.tab-nav button.selected {
    background: rgba(255,255,255,0.1) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* --- ANSWER SECTION --- */
#answer-section {
    background: rgba(0,0,0,0.2) !important;
    border-radius: 12px;
    padding: 1.5rem !important;
    border: 1px solid var(--border-glass);
}
#answer-section .markdown {
    font-size: 1.1rem;
    line-height: 1.7;
    color: var(--text-primary);
}
#answer-section strong {
    color: #60A5FA;
}

/* Footer Badge */
.calendar-badge {
    background: rgba(16, 185, 129, 0.15);
    color: var(--accent);
    padding: 0.6rem 1.2rem;
    border-radius: 100px;
    font-weight: 600;
    text-decoration: none;
    border: 1px solid rgba(16, 185, 129, 0.3);
    transition: all 0.2s;
}
.calendar-badge:hover {
    background: rgba(16, 185, 129, 0.25);
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
}
"""

with gr.Blocks(css=css, theme=gr.themes.Base(), title="Enterprise RAG") as demo:
    with gr.Column(elem_id="main-container"):
        # --- HERO ---
        gr.HTML("""
            <div id="header">
                <h1>ENTERPRISE RAG PLATFORM</h1>
                <p>Secure, Scalable, Agentic Document Intelligence for the Modern Enterprise.</p>
                <div style="margin-top: 3rem; margin-bottom: 6rem;" id="calendar-button">
                    <a href="https://cal.com" target="_blank" class="calendar-badge">
                        <span>📅</span> Book a 30-min Strategy Call
                    </a>
                </div>
            </div>
        """)

        with gr.Row(equal_height=True):  # Force Row to try to equalize height
            # --- LEFT: SETUP CARD ---
            with gr.Column(scale=4):
                with gr.Group(elem_classes="glass-card"):
                    gr.Markdown(
                        "### SELECT SAMPLE DOCUMENTS", elem_classes="card-header"
                    )
                    gr.Markdown(
                        "<span style='font-size: 0.8rem; opacity: 0.8; margin-bottom: 10px !important;'>_Choose a vertical to load pre-configured samples (Legal, FinOps)_</span>",
                        elem_classes="subtitle",
                    )

                    # Custom Tabs
                    with gr.Tabs():
                        with gr.Tab("⚖️ Legal"):
                            load_legal = gr.Button(
                                "Load Legal Samples", elem_classes="primary-btn"
                            )
                        with gr.Tab("🔬 Research"):
                            load_research = gr.Button(
                                "Load Research Samples", elem_classes="primary-btn"
                            )
                        with gr.Tab("💰 FinOps"):
                            load_finops = gr.Button(
                                "Load FinOps Samples", elem_classes="primary-btn"
                            )

                    load_status = gr.Markdown("", elem_classes="status-message")

                    # Visible Divider - Increased Opacity
                    gr.HTML(
                        '<div style="margin: 2rem 0; height: 1px; background: rgba(255,255,255,0.3);"></div>'
                    )

                    gr.Markdown("### OR UPLOAD FILES", elem_classes="card-header")
                    file_upload = gr.File(
                        file_types=[".pdf", ".docx", ".txt"],
                        show_label=False,
                        height=240,  # Increased height
                    )

                    # Spacer before Process button
                    gr.HTML('<div style="height: 1.5rem"></div>')

                    process_btn = gr.Button(
                        "Process Documents", elem_classes="primary-btn"
                    )
                    upload_status = gr.Markdown("")

                    # Spacer to fill height if needed
                    gr.HTML('<div style="flex-grow: 1;"></div>')

            # --- RIGHT: INTERACTION CARD ---
            with gr.Column(scale=6):
                with gr.Group(elem_classes="glass-card"):
                    gr.Markdown("### INTELLIGENT ANALYSIS", elem_classes="card-header")

                    # Question Input
                    question = gr.Textbox(
                        placeholder="Ask anything about your documents (e.g., 'What are the termination conditions?')...",
                        show_label=False,
                        lines=3,
                        elem_classes="gradio-textbox",
                    )

                    with gr.Row():
                        ask_btn = gr.Button(
                            "Analyze & Answer", elem_classes="primary-btn", scale=2
                        )

                    # Divider between Analyze and Quick Questions
                    gr.HTML(
                        '<div style="margin: 2rem 0; height: 1px; background: rgba(255,255,255,0.3);"></div>'
                    )

                    gr.Markdown(
                        "### QUICK SAMPLE QUESTIONS", elem_classes="card-header"
                    )
                    with gr.Row():
                        q1 = gr.Button("📋 Termination Terms", elem_classes="query-btn")
                        q2 = gr.Button("💰 Payment Summary", elem_classes="query-btn")
                    with gr.Row():
                        q3 = gr.Button("📊 Key Findings", elem_classes="query-btn")
                        q4 = gr.Button("⚠️ Risk Analysis", elem_classes="query-btn")

                    # Answer Output
                    gr.HTML('<div style="height: 2rem"></div>')
                    with gr.Group(elem_id="answer-section"):
                        gr.Markdown("### 🤖 Model Response", elem_classes="card-header")
                        answer = gr.Markdown("_AI analysis will appear here..._")

    # --- FOOTER ---
    with gr.Row(elem_id="footer-info"):
        gr.HTML("""
            <div style="text-align: center; color: var(--text-secondary); margin-top: 3rem; padding-bottom: 2rem; font-size: 0.9rem;">
                <p>🔒 <strong>Secure Environment</strong>: Documents processed locally & auto-deleted after 7 days.</p>
                <p style="margin-top: 0.5rem; opacity: 0.6;">© 2024 Enterprise RAG Platform. Licensed under MIT.</p>
            </div>
        """)

    # Event Wiring
    load_legal.click(fn=lambda: app.load_samples("Legal"), outputs=load_status)
    load_research.click(fn=lambda: app.load_samples("Research"), outputs=load_status)
    load_finops.click(fn=lambda: app.load_samples("FinOps"), outputs=load_status)

    process_btn.click(fn=app.process_file, inputs=file_upload, outputs=upload_status)

    q1.click(
        fn=lambda: f"**Query:** Termination Terms\n\n{app.ask('What are the termination conditions?')}",
        outputs=answer,
    )
    q2.click(
        fn=lambda: f"**Query:** Payment Summary\n\n{app.ask('Summarize payment terms')}",
        outputs=answer,
    )
    q3.click(
        fn=lambda: f"**Query:** Key Findings\n\n{app.ask('Summarize key findings')}",
        outputs=answer,
    )
    q4.click(
        fn=lambda: f"**Query:** Risk Analysis\n\n{app.ask('What are the key risks mentioned?')}",
        outputs=answer,
    )

    ask_btn.click(fn=app.ask, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch(share=False)
