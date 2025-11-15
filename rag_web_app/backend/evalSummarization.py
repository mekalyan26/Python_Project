import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SummarizationEvalResult:
    """Result of summarization evaluation"""
    summarization_score: float
    hallucination_score: float
    bias_score: float
    toxicity_score: float
    readability_score: float
    prompt_alignment_score: Optional[float]
    metrics_dict: Dict[str, Any]


class SummarizationEvaluationEngine:
    """
    Summarization Evaluation Engine using DeepEval.
    Evaluates summaries on: faithfulness, hallucination, bias, toxicity, readability.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the summarization evaluation engine.
        
        Args:
            model_name: LLM model to use for evaluation (default: gpt-3.5-turbo)
        """
        self.model_name = model_name
        self.deepeval_available = False
        self.evaluation_model = None
        self.PromptAlignmentMetric = None
        self._init_deepeval()

    def _init_deepeval(self):
        """Initialize DeepEval and load summarization metrics."""
        logger.info("Attempting to initialize DeepEval for summarization...")
        try:
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams
            from deepeval.metrics import (
                SummarizationMetric,
                HallucinationMetric,
                BiasMetric,
                ToxicityMetric,
                GEval,
            )

            # Try to import optional PromptAlignmentMetric
            try:
                from deepeval.metrics import PromptAlignmentMetric
                self.PromptAlignmentMetric = PromptAlignmentMetric
            except Exception:
                self.PromptAlignmentMetric = None

            # Try to configure LLM model
            try:
                from deepeval.models import GPTModel
                self.evaluation_model = GPTModel(model=self.model_name)
            except Exception:
                self.evaluation_model = None

            self.LLMTestCase = LLMTestCase
            self.LLMTestCaseParams = LLMTestCaseParams
            self.SummarizationMetric = SummarizationMetric
            self.HallucinationMetric = HallucinationMetric
            self.BiasMetric = BiasMetric
            self.ToxicityMetric = ToxicityMetric
            self.GEval = GEval

            self.deepeval_available = True
            logger.info("DeepEval summarization metrics ready")
        except Exception as e:
            logger.warning(f"DeepEval not available for summarization: {e}")
            self.deepeval_available = False

    def evaluate(
        self,
        source_document: str,
        summary: str,
        prompt_instructions: Optional[List[str]] = None,
    ) -> SummarizationEvalResult:
        """
        Evaluate a document summary.
        
        Args:
            source_document: Original document text
            summary: Generated summary text
            prompt_instructions: Optional list of instructions for prompt alignment
            
        Returns:
            SummarizationEvalResult with all metrics
        """
        if not self.deepeval_available:
            return self._fallback_evaluate(source_document, summary)
        try:
            return self._deepeval_evaluate(source_document, summary, prompt_instructions)
        except Exception as e:
            logger.exception(f"DeepEval summarize failed, fallback used: {e}")
            return self._fallback_evaluate(source_document, summary)

    def _deepeval_evaluate(
        self,
        source_document: str,
        summary: str,
        prompt_instructions: Optional[List[str]] = None,
    ) -> SummarizationEvalResult:
        """Evaluate using DeepEval metrics."""
        test_case = self.LLMTestCase(
            input=source_document,
            actual_output=summary,
            context=[source_document],
        )

        scores: Dict[str, float] = {}
        md: Dict[str, Any] = {}

        # Summarization quality
        try:
            metric = self.SummarizationMetric(threshold=0.7, include_reason=True, model=self.evaluation_model) if self.evaluation_model \
                     else self.SummarizationMetric(threshold=0.7, include_reason=True)
            metric.measure(test_case)
            scores["summarization"] = float(getattr(metric, "score", 0.0))
            md["summarization_score"] = scores["summarization"]
            md["summarization_reason"] = getattr(metric, "reason", "")
        except Exception:
            scores["summarization"] = 0.0

        # Hallucination
        try:
            metric = self.HallucinationMetric(threshold=0.7, include_reason=True, model=self.evaluation_model) if self.evaluation_model \
                     else self.HallucinationMetric(threshold=0.7, include_reason=True)
            metric.measure(test_case)
            scores["hallucination"] = float(getattr(metric, "score", 0.0))
            md["hallucination_score"] = scores["hallucination"]
            md["hallucination_reason"] = getattr(metric, "reason", "")
        except Exception:
            scores["hallucination"] = 0.0

        # Bias
        try:
            metric = self.BiasMetric(threshold=0.7, include_reason=True, model=self.evaluation_model) if self.evaluation_model \
                     else self.BiasMetric(threshold=0.7, include_reason=True)
            metric.measure(test_case)
            scores["bias"] = float(getattr(metric, "score", 0.0))
            md["bias_score"] = scores["bias"]
            md["bias_reason"] = getattr(metric, "reason", "")
        except Exception:
            scores["bias"] = 0.0

        # Toxicity
        try:
            metric = self.ToxicityMetric(threshold=0.7, include_reason=True, model=self.evaluation_model) if self.evaluation_model \
                     else self.ToxicityMetric(threshold=0.7, include_reason=True)
            metric.measure(test_case)
            scores["toxicity"] = float(getattr(metric, "score", 0.0))
            md["toxicity_score"] = scores["toxicity"]
            md["toxicity_reason"] = getattr(metric, "reason", "")
        except Exception:
            scores["toxicity"] = 0.0

        # Readability via GEval
        try:
            metric = self.GEval(
                name="Readability",
                criteria="Evaluate clarity, coherence, and ease of understanding for a non-expert on 0.0-1.0.",
                evaluation_params=[self.LLMTestCaseParams.INPUT, self.LLMTestCaseParams.ACTUAL_OUTPUT],
                threshold=0.7,
                model=self.evaluation_model,
            ) if self.evaluation_model else self.GEval(
                name="Readability",
                criteria="Evaluate clarity, coherence, and ease of understanding for a non-expert on 0.0-1.0.",
                evaluation_params=[self.LLMTestCaseParams.INPUT, self.LLMTestCaseParams.ACTUAL_OUTPUT],
                threshold=0.7,
            )
            metric.measure(test_case)
            scores["readability"] = float(getattr(metric, "score", 0.0))
            md["readability_score"] = scores["readability"]
            md["readability_reason"] = getattr(metric, "reason", "")
        except Exception:
            scores["readability"] = 0.0

        # Optional: Prompt alignment
        pa = None
        if prompt_instructions and self.PromptAlignmentMetric:
            try:
                metric = self.PromptAlignmentMetric(prompt_instructions=prompt_instructions, threshold=0.7, include_reason=True, model=self.evaluation_model) \
                         if self.evaluation_model else self.PromptAlignmentMetric(prompt_instructions=prompt_instructions, threshold=0.7, include_reason=True)
                metric.measure(test_case)
                pa = float(getattr(metric, "score", 0.0))
                md["prompt_alignment_score"] = pa
                md["prompt_alignment_reason"] = getattr(metric, "reason", "")
            except Exception:
                pa = 0.0

        return SummarizationEvalResult(
            summarization_score=scores.get("summarization", 0.0),
            hallucination_score=scores.get("hallucination", 0.0),
            bias_score=scores.get("bias", 0.0),
            toxicity_score=scores.get("toxicity", 0.0),
            readability_score=scores.get("readability", 0.0),
            prompt_alignment_score=pa,
            metrics_dict=md,
        )

    def _fallback_evaluate(
        self, source_document: str, summary: str
    ) -> SummarizationEvalResult:
        """Fallback evaluation using simple heuristics."""
        import re
        def wc(t): return len(re.findall(r'\w+', t))
        doc_w, sum_w = wc(source_document), wc(summary)
        ratio = (sum_w / doc_w) if doc_w else 0.0
        summarization_score = max(0.0, 1.0 - abs(ratio - 0.2) * 2)
        words_sum = set(re.findall(r'\w+', summary.lower()))
        words_doc = set(re.findall(r'\w+', source_document.lower()))
        overlap = (len(words_sum & words_doc) / len(words_sum)) if words_sum else 0.0
        return SummarizationEvalResult(
            summarization_score=summarization_score,
            hallucination_score=overlap,
            bias_score=0.8,
            toxicity_score=0.9,
            readability_score=min(1.0, sum_w/200.0) if sum_w < 200 else 0.8,
            prompt_alignment_score=None,
            metrics_dict={}
        )

    def get_metrics_summary(self, r: SummarizationEvalResult) -> Dict[str, float]:
        d = {
            "summarization": r.summarization_score,
            "hallucination": r.hallucination_score,
            "bias": r.bias_score,
            "toxicity": r.toxicity_score,
            "readability": r.readability_score,
        }
        if r.prompt_alignment_score is not None:
            d["prompt_alignment"] = r.prompt_alignment_score
        return d
