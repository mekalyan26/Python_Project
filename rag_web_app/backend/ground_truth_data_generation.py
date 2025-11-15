import os
import json
import logging
import re
from typing import List, Dict, Optional

# try to reuse your project's pdf utilities if present
try:
    from pdf_utils import extract_text_from_pdf
except Exception:
    extract_text_from_pdf = None

logger = logging.getLogger(__name__)


def _read_pdf_text(pdf_path: str) -> str:
    """Read text from PDF. Prefer project utility, fallback to PyPDF2."""
    if extract_text_from_pdf:
        try:
            return extract_text_from_pdf(pdf_path)
        except Exception as e:
            logger.exception("pdf_utils.extract_text_from_pdf failed, falling back: %s", e)

    try:
        # lightweight fallback using PyPDF2
        import PyPDF2
        text_parts = []
        with open(pdf_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for p in reader.pages:
                text_parts.append(p.extract_text() or "")
        return "\n\n".join(text_parts)
    except Exception:
        logger.exception("Failed to read PDF using PyPDF2")
        return ""


def _simple_synthesizer(text: str, max_q: int = 50) -> List[Dict[str, str]]:
    """
    Very small fallback QA generator:
    - split into sentences, pick candidate sentences
    - generate a short question from each by using the first 6-10 words
    - use the sentence as the answer and fact
    """
    # normalize & split
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # split to sentences using punctuation heuristic
    sents = re.split(r'(?<=[\.\?\!])\s+', text)
    # filter and keep reasonably sized sentences
    sents = [s.strip() for s in sents if 20 < len(s) < 800]
    results = []
    for i, sent in enumerate(sents[:max_q]):
        head = " ".join(sent.split()[:8])
        q = f"What is stated about \"{head}\"?"
        results.append({"question": q, "answer": sent, "fact": sent})
    return results


def _deepeval_synthesizer(text: str, num_questions: int = 50) -> Optional[List[Dict[str, str]]]:
    """
    Try to use DeepEval synthesizer API. Adapt if your installed deepeval API differs.
    """
    try:
        try:
            from deepeval.synthesizer import Synthesizer
        except Exception:
            try:
                from deepeval import Synthesizer
            except Exception:
                Synthesizer = None

        if Synthesizer is None:
            logger.debug("DeepEval Synthesizer not found")
            return None

        synth = Synthesizer()
        docs = [{"id": "doc0", "text": text}]
        
        # Try different method names and signatures
        for method_name in ("generate_from_documents", "synthesize_from_documents", "generate"):
            method = getattr(synth, method_name, None)
            if callable(method):
                logger.debug("Trying deepeval Synthesizer.%s", method_name)
                try:
                    # Try with num_questions first
                    try:
                        out = method(docs, num_questions=num_questions)
                    except TypeError:
                        # Try alternate signatures
                        try:
                            out = method(docs, n_questions=num_questions)
                        except TypeError:
                            out = method(docs)
                    
                    if not out:
                        logger.debug("Empty output from %s", method_name)
                        continue
                    
                    # Normalize output
                    results = []
                    for item in out:
                        if isinstance(item, dict):
                            q = item.get("question") or item.get("prompt") or item.get("q") or ""
                            a = item.get("answer") or item.get("response") or item.get("a") or ""
                            f = item.get("fact") or a
                            if q and a:
                                results.append({"question": q, "answer": a, "fact": f})
                        elif hasattr(item, "question"):
                            # object with attributes
                            results.append({
                                "question": getattr(item, "question", ""),
                                "answer": getattr(item, "answer", ""),
                                "fact": getattr(item, "fact", getattr(item, "answer", ""))
                            })
                    
                    if results:
                        logger.info("DeepEval produced %d goldens via %s", len(results), method_name)
                        return results
                except Exception as e:
                    logger.debug("Method %s failed: %s", method_name, e)
                    continue
        
        logger.debug("No compatible deepeval method found")
        return None
    except Exception:
        logger.exception("DeepEval synthesizer failed")
        return None


def generate_ground_truth_from_pdf(
    pdf_path: str,
    output_filename: Optional[str] = None,
    num_questions: int = 50,
    use_deepeval: bool = True,
) -> str:
    """
    Generate goldens from a PDF. Writes a JSON file next to the PDF and returns the path.
    Output JSON format: list of objects {question, answer, fact}
    """
    logger.info("Generating ground truth from PDF: %s", pdf_path)
    text = _read_pdf_text(pdf_path)
    if not text:
        raise RuntimeError("No text extracted from PDF")

    results = None
    if use_deepeval:
        results = _deepeval_synthesizer(text, num_questions=num_questions)
        if results:
            logger.info("DeepEval produced %d goldens", len(results))

    if not results:
        logger.info("Falling back to simple synthesizer")
        results = _simple_synthesizer(text, max_q=num_questions)

    # ensure output file path
    dirpath = os.path.dirname(pdf_path)
    base = output_filename or (os.path.splitext(os.path.basename(pdf_path))[0] + ".goldens.json")
    out_path = os.path.join(dirpath, base)

    # Write JSON
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        logger.info("Wrote groundtruth JSON to %s", out_path)
    except Exception:
        logger.exception("Failed to write groundtruth JSON")
        raise

    return out_path