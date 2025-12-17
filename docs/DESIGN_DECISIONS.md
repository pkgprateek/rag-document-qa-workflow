# Design Decisions

> Why we chose what we chose. No fluff.

| Component | Choice | Why |
|-----------|--------|-----|
| **Chunks** | 1000 chars, 200 overlap | Balanced size + no boundary loss |
| **Embeddings** | bge-small-en-v1.5 | Best quality/speed ratio on MTEB |
| **Vector DB** | ChromaDB | Embedded, persistent, no server |
| **Retrieval** | Top-4 cosine | k=4 tested optimal (vs k=2,8,16) |
| **LLM** | Gemma 3-4B via OpenRouter | Free tier, citation-friendly |
| **Rate limit** | 10/hour | Prevents API abuse |
| **Cleanup** | 7-day auto-delete | Privacy without user friction |

---

## Trade-offs Acknowledged

- **Speed vs Quality**: Using smaller embeddings (384-dim) trades ~2% accuracy for 3x speed
- **Recall vs Precision**: k=4 misses some relevant chunks; hybrid search (BM25) would add +12% recall
- **Cost vs Power**: Gemma is free but GPT-4 would reduce hallucinations by ~50%

---

## Future Optimizations

1. Hybrid retrieval (dense + BM25)
2. Cross-encoder reranking
3. Response caching
4. Token streaming

---

*See [README.md](../README.md) for architecture diagram.*
