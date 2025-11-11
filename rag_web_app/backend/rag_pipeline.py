import os
import re
import numpy as np
import faiss
import pickle
import torch
from typing import Dict, Any, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

logger = logging.getLogger(__name__)

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

    def _compute_contextual_metrics(self, answer: str, contexts: List[str], hits: List[Tuple[str, float, dict]]) -> Dict[str, float]:
        """Compute all metrics with detailed logging."""
        logger.debug("=== Starting Metrics Computation ===")
        logger.debug(f"Answer text ({len(answer)} chars): {answer[:100]}...")
        logger.debug(f"Number of contexts: {len(contexts)}")
        
        try:
            # Normalize texts
            def normalize_text(text: str) -> str:
                return re.sub(r'\s+', ' ', text.lower().strip())

            answer_norm = normalize_text(answer)
            context_text = " ".join(contexts)
            context_norm = normalize_text(context_text)
            logger.debug("Texts normalized")

            # Calculate embedding-based relevancy
            logger.debug("Computing embeddings...")
            embeddings = self.embed_model.encode(
                [answer_norm, context_norm], 
                convert_to_numpy=True
            )
            
            answer_emb = embeddings[0] / np.linalg.norm(embeddings[0])
            context_emb = embeddings[1] / np.linalg.norm(embeddings[1])
            
            answer_relevancy = float(np.dot(answer_emb, context_emb))
            logger.debug(f"Answer Relevancy Score: {answer_relevancy:.4f}")

            # Token-based metrics
            logger.debug("Computing token-based metrics...")
            answer_tokens = set(answer_norm.split())
            context_tokens = set(context_norm.split())
            common_tokens = answer_tokens & context_tokens
            
            precision = len(common_tokens) / len(answer_tokens) if answer_tokens else 0
            recall = len(common_tokens) / len(context_tokens) if context_tokens else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            logger.debug(f"Precision: {precision:.4f}")
            logger.debug(f"Recall: {recall:.4f}")
            logger.debug(f"F1 Score: {f1:.4f}")

            # RAGAS composite score
            ragas = sum([answer_relevancy, precision, recall, f1]) / 4
            logger.debug(f"RAGAS Score: {ragas:.4f}")

            metrics = {
                "answer_relevancy": round(answer_relevancy, 4),
                "faithfulness": round(precision, 4),
                "contextual_recall": round(recall, 4),
                "contextual_precision": round(precision, 4),
                "contextual_relevancy": round(f1, 4),
                "ragas": round(ragas, 4)
            }
            
            logger.info("Metrics computed successfully: %s", metrics)
            return metrics
                
        except Exception as e:
            logger.exception("Error computing metrics")
            return {
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,
                "contextual_recall": 0.0,
                "contextual_precision": 0.0,
                "contextual_relevancy": 0.0,
                "ragas": 0.0
            }

    def answer(self, query: str, reference_answer: Optional[str] = None, top_k: int = 4, max_new_tokens: int = 512) -> Dict[str, Any]:
        hits = self.retrieve(query, top_k=top_k)
        contexts = [c for c, _, _ in hits]
        prompt = self._format_rag_prompt(query, contexts)

        # Generation kwargs tuned to reduce verbosity / repetition
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic, less verbose
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", self.tokenizer.eos_token_id),
            return_full_text=False,  # prefer generated part only
        )

        out_item = self.generator(prompt, **gen_kwargs)[0]
        generated = out_item.get("generated_text") if isinstance(out_item, dict) else str(out_item)

        # Remove prompt if present and extract after "Answer:" if model used that token
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        answer = generated.split("Answer:")[-1].strip() if "Answer:" in generated else generated.strip()

        # Basic cleanup: collapse whitespace, remove non-printables, truncate
        import re
        answer = re.sub(r'\s+', ' ', answer)
        # trim trailing weird chars
        answer = re.sub(r'^[\W_]+|[\W_]+$', '', answer)
        # limit length for UI (adjust chars as needed)
        max_chars = 1500
        if len(answer) > max_chars:
            answer = answer[:max_chars].rsplit(' ', 1)[0] + '...'

        result = {
            "answer": answer,
            "sources": [{"id": m["id"], "score": float(s), "snippet": ctx[:400]} for ctx, s, m in hits],
        }

        # If reference is provided and deepeval is available, use it
        if reference_answer:
            metrics = self.evaluate_answer(answer, reference_answer)
            result["metrics"] = metrics or {}
        else:
            # compute lightweight contextual metrics from retrieved chunks
            result["metrics"] = self._compute_contextual_metrics(answer, contexts, hits)

        # Before returning, log the metrics
        print("Debug - Computed metrics:", result.get("metrics"))
        
        return result

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

    def evaluate_answer(self, generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """
        Compute evaluation metrics using embedding similarity and token overlap.
        Falls back to contextual metrics if deepeval is not available.
        """
        try:
            # Normalize text for comparison
            def normalize_text(text: str) -> str:
                return re.sub(r'\s+', ' ', text.lower().strip())

            gen_norm = normalize_text(generated_answer)
            ref_norm = normalize_text(reference_answer)

            # Get embeddings for similarity
            embeddings = self.embed_model.encode(
                [gen_norm, ref_norm], 
                convert_to_numpy=True
            )
            
            # Normalize vectors
            gen_emb = embeddings[0] / np.linalg.norm(embeddings[0])
            ref_emb = embeddings[1] / np.linalg.norm(embeddings[1])
            
            # Compute similarity score
            similarity = float(np.dot(gen_emb, ref_emb))

            # Token overlap metrics
            gen_tokens = set(gen_norm.split())
            ref_tokens = set(ref_norm.split())
            
            common = gen_tokens & ref_tokens
            precision = len(common) / len(gen_tokens) if gen_tokens else 0
            recall = len(common) / len(ref_tokens) if ref_tokens else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            metrics = {
                "answer_relevancy": round(similarity, 4),
                "faithfulness": round(precision, 4),
                "contextual_recall": round(recall, 4),
                "contextual_precision": round(precision, 4),
                "contextual_relevancy": round(f1, 4),
                "ragas": round((similarity + f1) / 2, 4)  # composite score
            }
            
            print(f"Debug - Evaluation metrics computed: {metrics}")
            return metrics

        except Exception as e:
            logging.exception("Evaluation failed")
            return {
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,
                "contextual_recall": 0.0,
                "contextual_precision": 0.0,
                "contextual_relevancy": 0.0,
                "ragas": 0.0
            }
