import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RAGEvalInput:
    """Input for RAG evaluation"""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


@dataclass
class RAGEvalResult:
    """Result of RAG evaluation"""
    answer_relevancy_score: float
    faithfulness_score: float
    contextual_recall_score: float
    contextual_precision_score: float
    ragas_score: float
    metrics_dict: Dict[str, float]


class RAGEvaluationEngine:
    """
    RAG Evaluation Engine using DeepEval.
    Evaluates RAG systems on multiple metrics: answer relevancy, faithfulness, 
    contextual recall, contextual precision, and RAGAS score.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the evaluation engine with DeepEval metrics.
        
        Args:
            model_name: LLM model to use for evaluation (default: gpt-3.5-turbo)
                       Can also use: gpt-4, gpt-4-turbo, claude-3-opus, etc.
        """
        self.model_name = model_name
        self.deepeval_available = False
        self.evaluation_model = None
        self.AnswerRelevancyMetric = None
        self.FaithfulnessMetric = None
        self.ContextualRecallMetric = None
        self.ContextualPrecisionMetric = None
        self.LLMTestCase = None
        self._init_deepeval()

    def _init_deepeval(self):
        """Initialize DeepEval and load metrics with LLM configuration."""
        logger.info("Attempting to initialize DeepEval...")
        try:
            # Import DeepEval components
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                FaithfulnessMetric,
                ContextualRecallMetric,
                ContextualPrecisionMetric,
            )
            from deepeval.test_case import LLMTestCase
            
            # Try to import and configure LLM model
            try:
                from deepeval.models import GPTModel
                logger.info(f"Using GPTModel with {self.model_name}")
                self.evaluation_model = GPTModel(model=self.model_name)
                logger.info(f"✓ GPTModel initialized: {self.evaluation_model}")
            except ImportError as e:
                logger.warning(f"GPTModel import failed: {e}")
                logger.warning("Metrics will use DeepEval default model (requires OPENAI_API_KEY)")
                self.evaluation_model = None
            except Exception as e:
                logger.warning(f"Failed to create GPTModel: {e}")
                logger.warning("Metrics will use DeepEval default model")
                self.evaluation_model = None

            self.AnswerRelevancyMetric = AnswerRelevancyMetric
            self.FaithfulnessMetric = FaithfulnessMetric
            self.ContextualRecallMetric = ContextualRecallMetric
            self.ContextualPrecisionMetric = ContextualPrecisionMetric
            self.LLMTestCase = LLMTestCase

            self.deepeval_available = True
            logger.info("✓ DeepEval metrics initialized successfully")
            logger.info(f"  - Model: {self.model_name}")
            logger.info(f"  - Evaluation model: {self.evaluation_model}")
            logger.info(f"  - AnswerRelevancyMetric: available")
            logger.info(f"  - FaithfulnessMetric: available")
            logger.info(f"  - ContextualRecallMetric: available")
            logger.info(f"  - ContextualPrecisionMetric: available")
            
        except ImportError as e:
            logger.warning(f"❌ DeepEval import failed: {e}")
            logger.warning("   Install with: pip install deepeval")
            logger.warning("   Also ensure OpenAI API key is set:")
            logger.warning("   PowerShell: $env:OPENAI_API_KEY='your-key'")
            logger.warning("   Or in .env file: OPENAI_API_KEY=your-key")
            self.deepeval_available = False
        except Exception as e:
            logger.exception(f"❌ Failed to initialize DeepEval: {e}")
            self.deepeval_available = False

    def evaluate(self, eval_input: RAGEvalInput) -> RAGEvalResult:
        """
        Evaluate a RAG system output.
        
        Args:
            eval_input: RAGEvalInput with question, answer, contexts, and optional ground_truth
            
        Returns:
            RAGEvalResult with all metrics
        """
        logger.info(f"evaluate() called. deepeval_available={self.deepeval_available}")
        
        if not self.deepeval_available:
            logger.warning("DeepEval not available, using fallback evaluation")
            return self._fallback_evaluate(eval_input)

        try:
            return self._deepeval_evaluate(eval_input)
        except Exception as e:
            logger.exception(f"DeepEval evaluation failed, falling back: {e}")
            return self._fallback_evaluate(eval_input)

    def _deepeval_evaluate(self, eval_input: RAGEvalInput) -> RAGEvalResult:
        """Evaluate using DeepEval metrics (following official docs pattern)."""
        logger.info(f"_deepeval_evaluate() called")
        logger.debug(f"  Question: {eval_input.question[:50]}...")
        logger.debug(f"  Answer: {eval_input.answer[:50]}...")
        logger.debug(f"  Contexts: {len(eval_input.contexts)} items")
        logger.debug(f"  Ground truth: {eval_input.ground_truth[:50] if eval_input.ground_truth else 'None'}...")

        if not self.LLMTestCase:
            logger.error("LLMTestCase not available!")
            raise RuntimeError("DeepEval LLMTestCase not initialized")

        # Create test case (as per DeepEval docs)
        test_case = self.LLMTestCase(
            input=eval_input.question,
            actual_output=eval_input.answer,
            retrieval_context=eval_input.contexts,
            expected_output=eval_input.ground_truth,
        )
        logger.debug(f"✓ Created LLMTestCase")

        metrics_dict = {}
        scores = {}

        # Define metrics with model parameter (key requirement from docs!)
        # See: https://deepeval.com/docs/getting-started-rag#define-metrics
        
        # 1. Answer Relevancy - measures how relevant answer is to question
        try:
            logger.info("  → Evaluating Answer Relevancy...")
            if self.evaluation_model:
                answer_relevancy = self.AnswerRelevancyMetric(
                    threshold=0.7,
                    model=self.evaluation_model,
                    include_reason=True
                )
            else:
                # Use default model (requires OPENAI_API_KEY)
                answer_relevancy = self.AnswerRelevancyMetric(
                    threshold=0.7,
                    include_reason=True
                )
            
            answer_relevancy.measure(test_case)
            score = getattr(answer_relevancy, 'score', 0.0)
            reason = getattr(answer_relevancy, 'reason', 'No reason provided')
            
            scores["answer_relevancy"] = score
            metrics_dict["answer_relevancy_score"] = score
            logger.info(f"    ✓ Answer Relevancy: {score:.4f}")
            logger.debug(f"    Reason: {reason}")
        except Exception as e:
            logger.exception(f"  ✗ Answer Relevancy metric failed: {e}")
            scores["answer_relevancy"] = 0.0

        # 2. Faithfulness - measures if answer is grounded in context
        try:
            logger.info("  → Evaluating Faithfulness...")
            if self.evaluation_model:
                faithfulness = self.FaithfulnessMetric(
                    threshold=0.7,
                    model=self.evaluation_model,
                    include_reason=True
                )
            else:
                faithfulness = self.FaithfulnessMetric(
                    threshold=0.7,
                    include_reason=True
                )
            
            faithfulness.measure(test_case)
            score = getattr(faithfulness, 'score', 0.0)
            reason = getattr(faithfulness, 'reason', 'No reason provided')
            
            scores["faithfulness"] = score
            metrics_dict["faithfulness_score"] = score
            logger.info(f"    ✓ Faithfulness: {score:.4f}")
            logger.debug(f"    Reason: {reason}")
        except Exception as e:
            logger.exception(f"  ✗ Faithfulness metric failed: {e}")
            scores["faithfulness"] = 0.0

        # 3. Contextual Recall - requires ground truth
        try:
            logger.info("  → Evaluating Contextual Recall...")
            if not eval_input.ground_truth:
                logger.warning("    ⚠️  No ground truth provided, skipping Contextual Recall")
                scores["contextual_recall"] = 0.0
            else:
                if self.evaluation_model:
                    contextual_recall = self.ContextualRecallMetric(
                        threshold=0.7,
                        model=self.evaluation_model,
                        include_reason=True
                    )
                else:
                    contextual_recall = self.ContextualRecallMetric(
                        threshold=0.7,
                        include_reason=True
                    )
                
                contextual_recall.measure(test_case)
                score = getattr(contextual_recall, 'score', 0.0)
                reason = getattr(contextual_recall, 'reason', 'No reason provided')
                
                scores["contextual_recall"] = score
                metrics_dict["contextual_recall_score"] = score
                logger.info(f"    ✓ Contextual Recall: {score:.4f}")
                logger.debug(f"    Reason: {reason}")
        except Exception as e:
            logger.exception(f"  ✗ Contextual Recall metric failed: {e}")
            scores["contextual_recall"] = 0.0

        # 4. Contextual Precision - requires ground truth
        try:
            logger.info("  → Evaluating Contextual Precision...")
            if not eval_input.ground_truth:
                logger.warning("    ⚠️  No ground truth provided, skipping Contextual Precision")
                scores["contextual_precision"] = 0.0
            else:
                if self.evaluation_model:
                    contextual_precision = self.ContextualPrecisionMetric(
                        threshold=0.7,
                        model=self.evaluation_model,
                        include_reason=True
                    )
                else:
                    contextual_precision = self.ContextualPrecisionMetric(
                        threshold=0.7,
                        include_reason=True
                    )
                
                contextual_precision.measure(test_case)
                score = getattr(contextual_precision, 'score', 0.0)
                reason = getattr(contextual_precision, 'reason', 'No reason provided')
                
                scores["contextual_precision"] = score
                metrics_dict["contextual_precision_score"] = score
                logger.info(f"    ✓ Contextual Precision: {score:.4f}")
                logger.debug(f"    Reason: {reason}")
        except Exception as e:
            logger.exception(f"  ✗ Contextual Precision metric failed: {e}")
            scores["contextual_precision"] = 0.0

        # 5. RAGAS Score - composite of all metrics
        ragas = sum(scores.values()) / len(scores) if scores else 0.0
        metrics_dict["ragas_score"] = ragas
        logger.info(f"  → RAGAS Score: {ragas:.4f}")

        result = RAGEvalResult(
            answer_relevancy_score=scores.get("answer_relevancy", 0.0),
            faithfulness_score=scores.get("faithfulness", 0.0),
            contextual_recall_score=scores.get("contextual_recall", 0.0),
            contextual_precision_score=scores.get("contextual_precision", 0.0),
            ragas_score=ragas,
            metrics_dict=metrics_dict,
        )

        logger.info(f"✓ DeepEval evaluation complete: {metrics_dict}")
        return result

    def _fallback_evaluate(self, eval_input: RAGEvalInput) -> RAGEvalResult:
        """Fallback evaluation using simple heuristics."""
        logger.info("Using fallback evaluation (heuristic-based)")

        import re
        import numpy as np

        def normalize_text(text: str) -> str:
            return re.sub(r'\s+', ' ', text.lower().strip())

        # Normalize texts
        answer_norm = normalize_text(eval_input.answer)
        context_text = " ".join(eval_input.contexts)
        context_norm = normalize_text(context_text)

        logger.debug(f"  Answer tokens: {len(answer_norm.split())}")
        logger.debug(f"  Context tokens: {len(context_norm.split())}")

        # Token-based metrics
        answer_tokens = set(answer_norm.split())
        context_tokens = set(context_norm.split())
        common_tokens = answer_tokens & context_tokens

        faithfulness = (
            len(common_tokens) / len(answer_tokens) if answer_tokens else 0.0
        )
        contextual_recall = (
            len(common_tokens) / len(context_tokens) if context_tokens else 0.0
        )
        contextual_precision = faithfulness

        # Embedding-based answer relevancy
        answer_relevancy = 0.0
        try:
            from sentence_transformers import SentenceTransformer

            logger.debug("  Computing embedding-based relevancy...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            emb = model.encode([answer_norm, context_norm])
            a_emb = emb[0] / np.linalg.norm(emb[0])
            c_emb = emb[1] / np.linalg.norm(emb[1])
            answer_relevancy = float(np.dot(a_emb, c_emb))
        except Exception as e:
            logger.debug(f"  Embedding-based evaluation failed: {e}")
            answer_relevancy = 0.5

        # RAGAS composite
        ragas = (
            answer_relevancy
            + faithfulness
            + contextual_recall
            + contextual_precision
        ) / 4

        metrics_dict = {
            "answer_relevancy_score": round(answer_relevancy, 4),
            "faithfulness_score": round(faithfulness, 4),
            "contextual_recall_score": round(contextual_recall, 4),
            "contextual_precision_score": round(contextual_precision, 4),
            "ragas_score": round(ragas, 4),
        }

        logger.info(f"✓ Fallback evaluation complete: {metrics_dict}")

        result = RAGEvalResult(
            answer_relevancy_score=answer_relevancy,
            faithfulness_score=faithfulness,
            contextual_recall_score=contextual_recall,
            contextual_precision_score=contextual_precision,
            ragas_score=ragas,
            metrics_dict=metrics_dict,
        )

        return result

    def batch_evaluate(
        self, eval_inputs: List[RAGEvalInput]
    ) -> List[RAGEvalResult]:
        """Evaluate multiple RAG outputs in batch."""
        logger.info(f"Starting batch evaluation of {len(eval_inputs)} items")
        results = []
        for i, eval_input in enumerate(eval_inputs):
            logger.debug(f"Evaluating item {i+1}/{len(eval_inputs)}")
            result = self.evaluate(eval_input)
            results.append(result)
        logger.info(f"✓ Batch evaluation complete: {len(results)} items evaluated")
        return results

    def get_metrics_summary(self, result: RAGEvalResult) -> Dict[str, float]:
        """Get a dictionary summary of evaluation results."""
        return {
            "answer_relevancy": result.answer_relevancy_score,
            "faithfulness": result.faithfulness_score,
            "contextual_recall": result.contextual_recall_score,
            "contextual_precision": result.contextual_precision_score,
            "ragas": result.ragas_score,
        }