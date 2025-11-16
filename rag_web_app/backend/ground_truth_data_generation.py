import json
import os
import re
import logging  # Add this import
from datetime import datetime
from typing import List, Optional, Dict
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.synthesizer import Synthesizer

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


async def generate_ground_truth(
    doc_id: str,
    num_questions: int = 5,
    question_type: str = "mixed",
    custom_prompt: Optional[str] = None
) -> dict:
    """
    Generate ground truth Q&A pairs using DeepEval Synthesizer or fallback
    
    Args:
        doc_id: Document identifier (can be path or ID)
        num_questions: Number of Q&A pairs to generate
        question_type: Type of questions (factual, analytical, mixed)
        custom_prompt: Optional custom prompt for generation
    
    Returns:
        dict with ground_truth_data, count, file_path
    """
    try:
        logger.info(f"📝 Generating {num_questions} ground truth samples for doc: {doc_id}")
        
        # Try to find the PDF file
        # First check if doc_id is a direct path
        pdf_path = None
        
        if os.path.isfile(doc_id):
            pdf_path = doc_id
        else:
            # Check in uploads directory
            uploads_dir = "uploads"
            if os.path.isdir(uploads_dir):
                # Try exact match
                potential_path = os.path.join(uploads_dir, doc_id)
                if os.path.isfile(potential_path):
                    pdf_path = potential_path
                else:
                    # Try adding .pdf extension
                    potential_path = os.path.join(uploads_dir, f"{doc_id}.pdf")
                    if os.path.isfile(potential_path):
                        pdf_path = potential_path
                    else:
                        # Search for any PDF with doc_id in name
                        for filename in os.listdir(uploads_dir):
                            if doc_id in filename and filename.endswith('.pdf'):
                                pdf_path = os.path.join(uploads_dir, filename)
                                break
        
        if not pdf_path or not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"Could not find PDF for doc_id: {doc_id}")
        
        logger.info(f"Found PDF at: {pdf_path}")
        
        # Extract text from PDF
        text = _read_pdf_text(pdf_path)
        if not text:
            raise RuntimeError("No text could be extracted from PDF")
        
        logger.info(f"Extracted {len(text)} characters from PDF")
        
        # Try DeepEval first
        goldens_list = None
        if question_type != "simple":  # Allow forcing simple mode
            goldens_list = _deepeval_synthesizer(text, num_questions=num_questions)
        
        # Fallback to simple synthesizer
        if not goldens_list:
            logger.info("Using fallback simple synthesizer")
            goldens_list = _simple_synthesizer(text, max_q=num_questions)
        
        if not goldens_list:
            raise RuntimeError("Failed to generate any ground truth samples")
        
        # Limit to requested number
        goldens_list = goldens_list[:num_questions]
        
        # Convert to standard format
        goldens_data = []
        for item in goldens_list:
            goldens_data.append({
                "input": item.get("question", ""),
                "expected_output": item.get("answer", ""),
                "actual_output": item.get("answer", ""),  # Same as expected initially
                "context": [item.get("fact", "")],
                "retrieval_context": [item.get("fact", "")],
            })
        
        # Save to JSON file
        output_dir = "data/ground_truth"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(datetime.now().timestamp())
        output_file = os.path.join(output_dir, f"ground_truth_{timestamp}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(goldens_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Generated {len(goldens_data)} samples, saved to {output_file}")
        
        return {
            "ground_truth_data": goldens_data,
            "count": len(goldens_data),
            "file_path": output_file,
            "doc_id": doc_id
        }
        
    except Exception as e:
        logger.exception("Ground truth generation failed")
        raise Exception(f"Ground truth generation failed: {str(e)}")