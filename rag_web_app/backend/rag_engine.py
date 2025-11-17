import os
import logging
from typing import List, Optional, Dict, Any
from llmConfig.llmConfigUtil import (
    LLMConfigUtil,
    get_default_llm_config,
    get_default_embedding_config
)

logger = logging.getLogger(__name__)


class RAGEngine:
    """Retrieval-Augmented Generation Engine with configurable LLM"""
    
    def __init__(self, index_dir: str = "data/faiss_index"):
        self.index_dir = index_dir
        self.index = None
        self.chunks = []
        
        # Load LLM configuration
        self.llm_config = get_default_llm_config()
        self.embedding_config = get_default_embedding_config()
        
        logger.info(f"✓ RAGEngine initialized with {self.llm_config.provider}/{self.llm_config.model}")
    
    def answer(
        self,
        question: str,
        reference_answer: Optional[str] = None,
        top_k: int = 4,
        max_new_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Generate answer using configurable LLM
        
        Args:
            question: User question
            reference_answer: Optional ground truth for evaluation
            top_k: Number of chunks to retrieve
            max_new_tokens: Max tokens for generated answer
            
        Returns:
            Dictionary with answer, contexts, and metrics
        """
        if self.index is None:
            raise ValueError("Index not loaded. Please upload a document first.")
        
        # 1. Retrieve relevant contexts
        contexts = self._retrieve_contexts(question, top_k)
        
        # 2. Build prompt with retrieved contexts
        context_text = "\n\n".join(contexts) if contexts else "No relevant context found."
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain the answer, say so."
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
            }
        ]
        
        # 3. Generate answer using configurable LLM
        try:
            logger.info(f"🤖 Generating answer using {self.llm_config.provider}/{self.llm_config.model}")
            
            answer = LLMConfigUtil.generate_completion(
                config=self.llm_config,
                messages=messages,
                max_tokens=max_new_tokens
            )
            
            logger.info(f"✓ Answer generated ({len(answer)} chars)")
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            answer = f"Error generating answer: {str(e)}"
        
        # 4. Return result
        return {
            "answer": answer,
            "contexts": contexts,
            "metrics": {
                "retrieved_chunks": len(contexts),
                "answer_length": len(answer),
                "model_used": f"{self.llm_config.provider}/{self.llm_config.model}"
            }
        }
    
    def _retrieve_contexts(self, question: str, top_k: int) -> List[str]:
        """
        Retrieve relevant contexts using configurable embeddings
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant text chunks
        """
        if not self.chunks:
            logger.warning("No chunks available for retrieval")
            return []
        
        try:
            # Generate question embedding
            question_embedding = LLMConfigUtil.generate_embeddings(
                config=self.embedding_config,
                texts=[question]
            )[0]
            
            # Search FAISS index
            import numpy as np
            D, I = self.index.search(np.array([question_embedding], dtype=np.float32), top_k)
            
            # Get corresponding chunks
            contexts = [self.chunks[i] for i in I[0] if i < len(self.chunks)]
            
            logger.info(f"✓ Retrieved {len(contexts)} contexts")
            return contexts
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return []
    
    def index_document(self, chunks: List[str]) -> bool:
        """
        Index document chunks using configurable embeddings
        
        Args:
            chunks: List of text chunks to index
            
        Returns:
            Success status
        """
        try:
            logger.info(f"📄 Indexing {len(chunks)} chunks...")
            
            # Generate embeddings for all chunks
            embeddings = LLMConfigUtil.generate_embeddings(
                config=self.embedding_config,
                texts=chunks
            )
            
            # Build FAISS index
            import faiss
            import numpy as np
            
            dimension = self.embedding_config.dimension
            embeddings_np = np.array(embeddings, dtype=np.float32)
            
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_np)
            self.chunks = chunks
            
            # Save index
            os.makedirs(self.index_dir, exist_ok=True)
            index_path = os.path.join(self.index_dir, "index.faiss")
            faiss.write_index(self.index, index_path)
            
            logger.info(f"✓ Indexed {len(chunks)} chunks successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Indexing failed: {e}")
            return False