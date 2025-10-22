import os
import faiss
import pickle
import torch
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

class RAGEngine:
    def __init__(self):
        # Embeddings
        self.embed_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        self.embed_dim = self.embed_model.get_sentence_embedding_dimension()

        # FAISS index (in-memory)
        self.index = None  # faiss.IndexFlatIP
        self.chunk_texts: List[str] = []
        self.chunk_meta: List[Dict[str, Any]] = []
        self.full_text: str = ""

        # Local LLM
        self._init_llm()

    def _init_llm(self):
        # Keep defaults simple and CPU-friendly
        kwargs = {}
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        else:
            # CPU path
            kwargs["torch_dtype"] = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(DEFAULT_LLM_MODEL, **kwargs)

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # device_map handled above via model kwargs
        )

    # ---------- Indexing ----------
    def _ensure_index(self):
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embed_dim)

    def _normalize(self, vectors):
        # cosine sim via inner product on normalized vectors
        import numpy as np
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def build_index(self, chunks: List[str]):
        import numpy as np
        self._ensure_index()
        self.chunk_texts = chunks
        self.chunk_meta = [{"id": i} for i in range(len(chunks))]

        embeddings = self.embed_model.encode(chunks, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
        embeddings = self._normalize(embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(embeddings)

    # ---------- Retrieval ----------
    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[str, float, Dict[str, Any]]]:
        if self.index is None or len(self.chunk_texts) == 0:
            return []

        import numpy as np
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        q_emb = self._normalize(q_emb).astype("float32")

        scores, idxs = self.index.search(q_emb, top_k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append((self.chunk_texts[idx], float(score), self.chunk_meta[idx]))
        return out

    # ---------- Generation ----------
    def _format_rag_prompt(self, query: str, contexts: List[str]) -> str:
        context_block = "\n\n---\n\n".join(contexts)
        prompt = (
            "You are a helpful assistant. Use ONLY the provided context to answer.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context_block}\n\n"
            f"User question: {query}\n\n"
            "Answer:"
        )
        return prompt

    def _format_summarize_prompt(self, text: str) -> str:
        # keep it small for tiny models by truncating
        max_chars = 12000  # adjust based on model capacity
        text = text[:max_chars]
        prompt = (
            "Summarize the following document for a technical audience. "
            "Use concise bullet points and a brief overview paragraph.\n\n"
            f"Document:\n{text}\n\nSummary:"
        )
        return prompt

    def answer(self, query: str, top_k: int = 4, max_new_tokens: int = 512) -> Dict[str, Any]:
        hits = self.retrieve(query, top_k=top_k)
        contexts = [c for c, _, _ in hits]
        prompt = self._format_rag_prompt(query, contexts)

        # tiny models benefit from lower max tokens
        out = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        # Strip the prompt to keep only the answer part (simple heuristic)
        answer = out.split("Answer:")[-1].strip() if "Answer:" in out else out

        return {
            "answer": answer,
            "sources": [
                {"id": m["id"], "score": s, "snippet": ctx[:400]} for ctx, s, m in hits
            ],
        }

    def summarize(self, max_new_tokens: int = 600) -> str:
        if not self.full_text.strip():
            return "No document loaded."
        prompt = self._format_summarize_prompt(self.full_text)
        out = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        summary = out.split("Summary:")[-1].strip() if "Summary:" in out else out
        return summary

    # ---------- Persistence (optional) ----------
    def save_state(self, dir_path: str):
        """Optional utility if you want to persist across runs."""
        import os, numpy as np
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "chunks.pkl"), "wb") as f:
            pickle.dump({"chunks": self.chunk_texts, "meta": self.chunk_meta, "full_text": self.full_text}, f)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(dir_path, "faiss.index"))

    def load_state(self, dir_path: str):
        """Optional utility to restore state."""
        import os
        chunks_pkl = os.path.join(dir_path, "chunks.pkl")
        index_path = os.path.join(dir_path, "faiss.index")
        if os.path.exists(chunks_pkl) and os.path.exists(index_path):
            with open(chunks_pkl, "rb") as f:
                data = pickle.load(f)
            self.chunk_texts = data["chunks"]
            self.chunk_meta = data["meta"]
            self.full_text = data.get("full_text", "")
            self.index = faiss.read_index(index_path)
