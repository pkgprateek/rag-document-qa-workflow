---
title: Enterprise RAG Platform
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app/main.py
pinned: false
license: mit
short_description: Document intelligence for Legal, Research, FinOps
full_width: true
---

# Enterprise RAG Platform

**Turn documents into answers. Instantly.**

Upload contracts, research papers, or financial reports → Ask questions → Get cited answers in seconds.

---

## ✨ What's New

- **Multi-document upload** — Process multiple files at once
- **Streaming answers** — Watch responses generate in real-time
- **Thinking indicator** — See "🔍 Analyzing documents..." before streaming starts

---

## How It Works

```
📄 Upload → ✂️ Chunk → 🧠 Embed → 💬 Ask → ✨ Cited Answer
```

**3 steps**: Upload your documents → Ask questions → Get answers with page citations.

---

## Try It Now

1. **Select a vertical** — Legal, Research, or FinOps samples pre-loaded
2. **Or upload your own** — PDF, DOCX, TXT supported (batch upload enabled)
3. **Ask anything** — Natural language questions
4. **Get streaming answers** — Watch the AI think and respond in real-time

No signup required. Documents auto-deleted after 7 days.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-upload** | Upload multiple files at once |
| **Streaming** | Real-time token-by-token answers |
| **Citations** | Every answer links to source + page |
| **3 AI models** | GPT-OSS 120B, Llama 3.3, Gemma 3 |
| **Privacy** | Session isolation, 7-day auto-delete |

---

## Run Locally

```bash
git clone https://github.com/pkgprateek/rag-document-qa-workflow.git
cd rag-document-qa-workflow
echo "GROQ_API_KEY=your_key" > .env
docker compose up
# → http://localhost:7860
```

**API Keys:** [Groq](https://console.groq.com/keys) (Required) · [OpenRouter](https://openrouter.ai/keys) (Optional)

---

## 🔒 Privacy

- Documents processed locally
- Session-isolated storage
- Auto-deleted after 7 days
- Never used for training

---

## Enterprise Pilots

**2-week paid pilots** for teams ready to deploy RAG on their infrastructure.

📅 [Book discovery call](https://cal.com/prateekgoel/30m-discovery-call)

---

**Built by [Prateek Kumar Goel](https://github.com/pkgprateek)** · MIT License
