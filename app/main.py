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
        """Load sample documents with live progress updates"""
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
            total_chunks = 0
            for idx, path in enumerate(samples[vertical], 1):
                if os.path.exists(path):
                    yield f"Loading document {idx}/{len(samples[vertical])}..."
                    chunks = self.processor.process_txt(path)

                    yield f"Creating smart chunks ({len(chunks)} chunks)..."
                    self.rag_pipeline.add_documents(chunks, is_sample=True)

                    yield f"Building search index..."
                    self.loaded_documents.append(os.path.basename(path))
                    total_chunks += len(chunks)

            yield f"✓ Success! Loaded {len(samples[vertical])} documents ({total_chunks} searchable chunks)"
        except Exception as e:
            yield f"❌ Error: {str(e)}"

    def process_file(self, file):
        """Process uploaded file with live progress updates"""
        if not file:
            yield "⚠️ Please upload a file"
            return

        try:
            filename = os.path.basename(file.name)
            yield f"Processing {filename}..."

            ext = os.path.splitext(file.name)[1].lower()
            if ext == ".pdf":
                chunks = self.processor.process_pdf(file.name)
            elif ext == ".txt":
                chunks = self.processor.process_txt(file.name)
            elif ext == ".docx":
                chunks = self.processor.process_docx(file.name)
            else:
                yield "❌ Unsupported format. Please upload PDF, DOCX, or TXT files."
                return

            yield f"✂️ Created {len(chunks)} smart chunks..."

            yield f"Building search index (securing with AES-256)..."
            self.rag_pipeline.add_documents(chunks, is_sample=False)
            self.loaded_documents.append(filename)

            yield f"✓ Success! {filename} ready for questions ({len(chunks)} searchable chunks)"
        except Exception as e:
            yield f"❌ Error: {str(e)}. Please try again or contact support."

    def switch_model(self, model_choice):
        """Handle model switching from UI radio button"""
        # Map UI choices to model keys
        model_map = {
            "GPT-OSS 120B (OpenAI) - Default": "gpt-oss-120b",
            "Llama 3.3 70B (Meta)": "llama-3.3-70b",
            "Gemma 3 27B (Google)": "gemma-3-27b",
        }

        model_key = model_map.get(model_choice)
        if not model_key:
            return f"❌ Invalid model selection"

        try:
            display_name = self.rag_pipeline.switch_model(model_key)
            return f"✓ Switched to {display_name}"
        except Exception as e:
            return f"❌ Error switching model: {str(e)}"

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
    /* Material Design Color Palette */
    --primary-gradient: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
    --cta-discovery-gradient: linear-gradient(135deg, #00C853 0%, #00A152 100%);
    --surface-dark: #0B0F19;
    --surface-glass: rgba(17, 24, 39, 0.7);
    --border-glass: rgba(255, 255, 255, 0.08); 
    --text-primary: #F9FAFB;
    --text-secondary: #9CA3AF;
    --accent: #2196F3;
    --accent-hover: #1976D2;
    --cta-discovery: #00C853;
    
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
    padding: 2rem 2rem 1.5rem 2rem !important; /* Reduced bottom padding */
    margin-bottom: 2rem !important;
    box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5) !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Prevent left column from expanding - constrain height and hide scrollbar */
.gradio-row > .gradio-column:first-child .glass-card {
    max-height: none;
    overflow: visible;
}

/* Hide all scrollbars in main container */
.gradio-container {
    overflow-x: hidden !important;
}

body {
    overflow-x: hidden !important;
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
    background-color: rgba(0, 0, 0, 0.15) !important;
    border: 2px dashed rgba(255, 255, 255, 0.3) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

.gradio-file:hover {
    background-color: rgba(0, 0, 0, 0.2) !important;
    border-color: var(--accent) !important;
}

.gradio-dropdown:hover, .gradio-textbox textarea:hover {
    border-color: var(--accent) !important;
}

/* Primary Button */
.primary-btn {
    background: var(--primary-gradient) !important;
    border: 1px solid rgba(33, 150, 243, 0.3) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 1rem !important;
    border-radius: 10px !important;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(33, 150, 243, 0.25);
    margin-top: 0 !important;
}
.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 0 1px rgba(33, 150, 243, 0.5), 0 8px 25px rgba(33, 150, 243, 0.4);
    border-color: rgba(33, 150, 243, 0.6) !important;
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

/* Calendar/Discovery Badge - Professional Green CTA */
.calendar-badge {
    background: linear-gradient(135deg, #00C853 0%, #00A152 100%) !important;
    color: white;
    padding: 0.75rem 1.6rem;
    border-radius: 100px;
    font-weight: 600;
    text-decoration: none;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(0, 200, 83, 0.35);
    display: inline-block;
}
.calendar-badge:hover {
    background: linear-gradient(135deg, #00A152 0%, #00853E 100%) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0, 200, 83, 0.5);
}
.calendar-badge span {
    font-size: 1.1rem;
    margin-right: 0.3rem;
}

/* --- MODEL SELECTOR --- */
.model-selector {
    background: rgba(0, 0, 0, 0.15) !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    margin-bottom: 1rem !important;
    border: 1px solid var(--border-glass) !important;
}

.model-selector label {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--border-glass) !important;
    padding: 0.5rem 0.75rem !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
    margin: 0.2rem 0 !important;
    display: block !important;
    font-size: 0.875rem !important;
}

.model-selector label:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: var(--accent) !important;
    transform: translateX(3px) !important;
}

.model-selector input:checked + label {
    background: var(--primary-gradient) !important;
    border-color: transparent !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 12px rgba(33, 150, 243, 0.4) !important;
}

.model-status {
    font-size: 0.8rem;
    color: var(--text-secondary);
    padding: 0.25rem 0.5rem;
    margin-top: 0.1rem;
}

/* --- SECURITY BADGE --- */
.security-badge {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px;
    padding: 0.5rem 0.8rem;
    margin-top: 0.5rem;
    margin-bottom: 0 !important;
    transition: all 0.3s ease;
}

.security-badge:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 0 15px rgba(100, 100, 100, 0.2);
}

.badge-icon {
    font-size: 1.3rem;
    line-height: 1;
    opacity: 0.9;
}

.badge-content {
    flex: 1;
}

.badge-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.15rem;
}

.badge-subtitle {
    font-size: 0.7rem;
    color: var(--text-secondary);
    opacity: 0.7;
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
                    <a href="https://cal.com/prateekgoel/30m-discovery-call" target="_blank" class="calendar-badge">
                        <span>📅</span> Book 30m Discovery Call
                    </a>
                </div>
            </div>
        """)

        with gr.Row(equal_height=False):
            # --- LEFT: SETUP CARD (45%) ---
            with gr.Column(scale=9):
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

                    # Horizontal divider - more visible
                    gr.HTML(
                        '<div style="margin: 1rem 0; height: 2px; background: rgba(255,255,255,0.2); border-radius: 1px;"></div>'
                    )

                    gr.Markdown("### OR UPLOAD DOCUMENTS", elem_classes="card-header")
                    file_upload = gr.File(
                        file_types=[".pdf", ".docx", ".txt"],
                        show_label=True,
                        height=240,  # Increased height
                    )

                    # Security Badge
                    gr.HTML("""
                        <div class="security-badge">
                            <div class="badge-icon">🔒</div>
                            <div class="badge-content">
                                <div class="badge-title">AES-256 Encrypted</div>
                                <div class="badge-subtitle">Processed locally • Auto-deleted in 7 days</div>
                            </div>
                        </div>
                    """)

                    process_btn = gr.Button(
                        "Process Documents", elem_classes="primary-btn"
                    )
                    upload_status = gr.Markdown("")

                    # Divider
                    gr.HTML(
                        '<div style="margin: 1rem 0; height: 1px; background: rgba(255,255,255,0.15);"></div>'
                    )

                    # Model Selector (Compact)
                    gr.Markdown("**🤖 Choose AI Model**", elem_classes="card-subheader")
                    model_selector = gr.Radio(
                        choices=[
                            "GPT-OSS 120B (OpenAI) - Default",
                            "Llama 3.3 70B (Meta)",
                            "Gemma 3 27B (Google)",
                        ],
                        value="GPT-OSS 120B (OpenAI) - Default",
                        elem_classes="model-selector",
                        show_label=False,
                    )
                    model_status = gr.Markdown(
                        "_GPT-OSS 120B active_",
                        elem_classes="model-status",
                    )

            # --- RIGHT: INTERACTION CARD (55%) ---
            with gr.Column(scale=11):
                with gr.Group(elem_classes="glass-card"):
                    gr.Markdown("### ASK ANYTHING", elem_classes="card-header")

                    # Question Input
                    question = gr.Textbox(
                        placeholder="Ask anything about your documents (e.g., 'What are the termination conditions?')...",
                        show_label=False,
                        lines=3,
                        elem_classes="gradio-textbox",
                    )

                    # Small spacing before action button
                    gr.HTML('<div style="height: 0.50rem"></div>')

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

    # Event Wiring with live updates (generators)
    load_legal.click(fn=lambda: app.load_samples("Legal"), outputs=load_status)
    load_research.click(fn=lambda: app.load_samples("Research"), outputs=load_status)
    load_finops.click(fn=lambda: app.load_samples("FinOps"), outputs=load_status)

    process_btn.click(fn=app.process_file, inputs=file_upload, outputs=upload_status)

    # Model switching
    model_selector.change(
        fn=app.switch_model, inputs=model_selector, outputs=model_status
    )

    q1.click(
        fn=lambda: app.ask("What are the termination conditions?"),
        outputs=answer,
    )
    q2.click(
        fn=lambda: app.ask("Summarize payment terms"),
        outputs=answer,
    )
    q3.click(
        fn=lambda: app.ask("Summarize key findings"),
        outputs=answer,
    )
    q4.click(
        fn=lambda: app.ask("What are the key risks mentioned?"),
        outputs=answer,
    )

    ask_btn.click(fn=app.ask, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch(share=False)
