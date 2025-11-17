import os, re, heapq, logging
import json  # added
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse  # added
from pydantic import BaseModel  # added
import logging
import os

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from llmConfig.llmConfigUtil import get_default_llm_config, get_default_embedding_config

llm_cfg = get_default_llm_config()
emb_cfg = get_default_embedding_config()
logger.info(f"🚀 RAG server starting | AI_STACK={os.getenv('AI_STACK','openai')}")
logger.info(f"   LLM: {llm_cfg.provider.value}/{llm_cfg.model}")
logger.info(f"   EMB: {emb_cfg.provider.value}/{emb_cfg.model} (dim={emb_cfg.dimension})")

# In-memory store of last uploaded document text and chunks
DOCUMENT_TEXT: str = ""
DOCUMENT_CHUNKS: List[str] = []

# Import project modules with error handling
try:
    from pdf_utils import extract_text_from_pdf, chunk_text
    logger.info("✓ pdf_utils imported")
except Exception as e:
    logger.error(f"Failed to import pdf_utils: {e}")
    raise

try:
    from rag_pipeline import RAGEngine
    logger.info("✓ rag_pipeline imported")
except Exception as e:
    logger.error(f"Failed to import rag_pipeline: {e}")
    raise

try:
    from ground_truth_data_generation import generate_ground_truth_from_pdf
    logger.info("✓ ground_truth_data_generation imported")
except Exception as e:
    logger.error(f"Failed to import ground_truth_data_generation: {e}")
    raise

try:
    from evalEngine import RAGEvaluationEngine, RAGEvalInput
    logger.info("✓ evalEngine imported")
except Exception as e:
    logger.error(f"Failed to import evalEngine: {e}")
    raise

from evalSummarization import SummarizationEvaluationEngine

# Create FastAPI app
app = FastAPI(title="Local RAG Backend", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines with error handling
logger.info("Initializing RAG engine...")
try:
    rag = RAGEngine()
    logger.info("✓ RAG engine initialized")
except Exception as e:
    logger.error(f"Failed to initialize RAG engine: {e}")
    rag = None

logger.info("Initializing Evaluation engine...")
try:
    eval_engine = RAGEvaluationEngine()
    logger.info("✓ Evaluation engine initialized")
except Exception as e:
    logger.error(f"Failed to initialize Evaluation engine: {e}")
    eval_engine = None

# Initialize summarization evaluation engine
try:
    summarization_eval_engine = SummarizationEvaluationEngine(model_name="gpt-3.5-turbo")
except Exception as e:
    logger.warning(f"Summarization eval engine init failed: {e}")
    summarization_eval_engine = None

def simple_extractive_summary(text: str, max_sentences: int = 5) -> str:
    """
    Lightweight extractive summarizer (no external deps).
    Scores sentences by term frequency (stopword-filtered).
    """
    if not text:
        return ""
    # Split sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # Tokenize + frequency
    words = re.findall(r'\w+', text.lower())
    stop = set("""a an the and or but if into in on at of to from for with without is are was were be been being this that those these it its as by we you they he she i me my our your their them his her""".split())
    freq = {}
    for w in words:
        if w in stop: continue
        freq[w] = freq.get(w, 0) + 1

    # Score sentences
    scores = []
    for i, s in enumerate(sentences):
        tokens = re.findall(r'\w+', s.lower())
        score = sum(freq.get(t, 0) for t in tokens) / (len(tokens) + 1)
        scores.append((score, i, s))

    best = heapq.nlargest(max_sentences, scores)
    best_sorted = [s for _, _, s in sorted(best, key=lambda x: x[1])]
    return " ".join(best_sorted)

# Request models
class AskRequest(BaseModel):
    question: str
    reference_answer: Optional[str] = None
    top_k: Optional[int] = 4  # Add this
    max_new_tokens: Optional[int] = 512  # Add this

class GroundTruthRequest(BaseModel):
    doc_id: str
    num_questions: int = 5
    question_type: str = "mixed"  # factual, analytical, mixed
    custom_prompt: Optional[str] = None

# Health check
@app.get("/health")
def health():
    return {"status": "ok", "rag": "ready" if rag else "not_ready"}

# Upload endpoint
@app.post("/upload")
async def upload(file: UploadFile = File(...), chunk_size: int = Form(900), chunk_overlap: int = Form(200)):
    try:
        from pdf_utils import extract_text_from_pdf, chunk_text
        raw = await file.read()
        # Save temp file -> extract text
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        text = extract_text_from_pdf(tmp_path) or ""
        chunks = chunk_text(text, chunk_size, chunk_overlap) if text else []

        # Store globally for summarize
        global DOCUMENT_TEXT, DOCUMENT_CHUNKS
        DOCUMENT_TEXT = text
        DOCUMENT_CHUNKS = chunks

        return {"ok": True, "message": f"Indexed {len(chunks)} chunks", "chunks": len(chunks)}
    except Exception as e:
        logger.exception("Upload failed")
        return {"ok": False, "message": str(e)}

def _all_zero(d: Optional[dict]) -> bool:
    """True if no metrics or all numeric values are 0.0"""
    if not isinstance(d, dict) or not d:
        return True
    has_any = False
    for v in d.values():
        if isinstance(v, (int, float)):
            has_any = True
            if float(v) != 0.0:
                return False
    return has_any  # True only if had numeric keys and all were 0.0

def _compute_summarization_fallback_metrics(source: str, summary: str, contexts: List[str]) -> dict:
    """Heuristic summarization metrics in [0,1] without external deps."""
    import re

    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    def tokens(text: str) -> List[str]:
        stop = set("""a an the and or but if into in on at of to from for with without is are was were be been being this that those these it its as by we you they he she i me my our your their them his her""".split())
        toks = re.findall(r"\w+", text.lower())
        return [t for t in toks if t not in stop]

    def syllable_count(word: str) -> int:
        # simple heuristic syllable count
        w = word.lower()
        w = re.sub(r'[^a-z]', '', w)
        if not w:
            return 0
        groups = re.findall(r'[aeiouy]+', w)
        count = len(groups)
        if w.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    def readability_score(text: str) -> float:
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        sents = [s for s in sents if s.strip()]
        words = re.findall(r'\w+', text)
        if not words or not sents:
            return 0.5
        syllables = sum(syllable_count(w) for w in words)
        asl = len(words) / max(1, len(sents))          # avg sentence length
        asw = syllables / max(1, len(words))           # avg syllables per word
        # Flesch Reading Ease
        fre = 206.835 - 1.015 * asl - 84.6 * asw       # typical range ~0..100
        fre = max(0.0, min(100.0, fre))
        return fre / 100.0

    src = normalize(source or "")
    summ = normalize(summary or "")
    ctx = normalize(" ".join(contexts or []))

    src_t = set(tokens(src))
    sum_t = set(tokens(summ))
    ctx_t = set(tokens(ctx)) if ctx else src_t

    if not sum_t:
        return {
            "summarization": 0.0,
            "hallucination": 0.0,
            "bias": 1.0,
            "toxicity": 1.0,
            "readability": 0.0,
        }

    # Faithfulness proxy = overlap with source/contexts
    overlap = len(sum_t & ctx_t) / max(1, len(sum_t))
    coverage = len(sum_t & src_t) / max(1, len(src_t))
    readability = readability_score(summary)

    # Scores in [0,1]
    faithfulness = overlap
    summarization_quality = 0.5 * faithfulness + 0.5 * readability
    bias = 1.0
    toxicity = 1.0

    return {
        "summarization": round(float(summarization_quality), 4),
        "hallucination": round(float(faithfulness), 4),  # higher = less hallucination
        "bias": round(bias, 4),
        "toxicity": round(toxicity, 4),
        "readability": round(float(readability), 4),
    }

# Ask endpoint
@app.post("/ask")
def ask(req: AskRequest):
    if rag is None:
        return {"ok": False, "message": "RAG engine not initialized"}

    logger.debug(f"Processing question: {req.question[:100]}...")
    try:
        result = rag.answer(
            req.question,
            reference_answer=req.reference_answer,
            top_k=req.top_k,  # Now uses the attribute from the model
            max_new_tokens=req.max_new_tokens,  # Now uses the attribute from the model
        )

        deepeval_metrics = {}
        fallback_eval_metrics = {}

        logger.info(f"eval_engine available: {eval_engine is not None}")
        logger.info(f"Reference answer provided: {req.reference_answer is not None}")

        if eval_engine:
            try:
                eval_input = RAGEvalInput(
                    question=req.question,
                    answer=result.get("answer", ""),
                    contexts=result.get("contexts", []),
                    ground_truth=req.reference_answer
                )
                logger.info("✓ RAGEvalInput created")
            except Exception as e:
                logger.warning(f"Failed to build RAGEvalInput: {e}")
                eval_input = None

            # Always compute fallback metrics (deterministic heuristic)
            if eval_input:
                try:
                    logger.info("Computing fallback metrics...")
                    fb_result = eval_engine._fallback_evaluate(eval_input)
                    fallback_eval_metrics = eval_engine.get_metrics_summary(fb_result)
                    logger.info(f"✓ Fallback metrics computed: {fallback_eval_metrics}")
                except Exception as e:
                    logger.warning(f"Fallback evaluation failed: {e}")
                    fallback_eval_metrics = {}

            # Compute DeepEval metrics if available (best-effort)
            if eval_input:
                logger.info(f"DeepEval available: {getattr(eval_engine, 'deepeval_available', False)}")
                if getattr(eval_engine, "deepeval_available", False):
                    try:
                        logger.info("Computing DeepEval metrics...")
                        if hasattr(eval_engine, "_deepeval_evaluate"):
                            de_result = eval_engine._deepeval_evaluate(eval_input)
                            deepeval_metrics = eval_engine.get_metrics_summary(de_result)
                        else:
                            de_result = eval_engine.evaluate(eval_input)
                            deepeval_metrics = eval_engine.get_metrics_summary(de_result)
                        logger.info(f"✓ DeepEval metrics computed: {deepeval_metrics}")
                    except Exception as e:
                        logger.exception(f"DeepEval evaluation failed: {e}")
                        logger.warning("Falling back to heuristic metrics for DeepEval panel")
                        deepeval_metrics = fallback_eval_metrics
                else:
                    logger.warning("⚠️  DeepEval not available - using fallback for DeepEval panel")
                    deepeval_metrics = fallback_eval_metrics

            # NEW: if DeepEval produced non-informative metrics, fall back to heuristic values
            if _all_zero(deepeval_metrics):
                logger.warning("DeepEval metrics are all zero; using fallback metrics for UI")
                deepeval_metrics = fallback_eval_metrics

        return {
            "ok": True,
            "answer": result.get("answer"),
            "contexts": result.get("contexts", []),
            "metrics": result.get("metrics", {}),
            "deepeval_metrics": deepeval_metrics,
            "fallback_eval_metrics": fallback_eval_metrics,
        }
    except Exception as e:
        logger.exception("Ask endpoint failed")
        return {"ok": False, "message": str(e)}

# Summarize endpoint
@app.get("/summarize")
def summarize():
    try:
        if not DOCUMENT_TEXT:
            return {"ok": False, "message": "No document uploaded yet"}

        # 1) Produce a summary (fallback extractive)
        summary = simple_extractive_summary(DOCUMENT_TEXT, max_sentences=5)

        # 2) Compute summarization metrics (DeepEval or fallback)
        summarization_metrics = {}
        if summarization_eval_engine:
            try:
                eval_result = summarization_eval_engine.evaluate(
                    source_document=DOCUMENT_TEXT,
                    summary=summary,
                )
                summarization_metrics = summarization_eval_engine.get_metrics_summary(eval_result)
            except Exception as e:
                logger.warning(f"Summarization eval failed: {e}")

        # NEW: fallback if metrics missing or all zeros
        if _all_zero(summarization_metrics):
            logger.info("Using heuristic fallback for summarization metrics")
            summarization_metrics = _compute_summarization_fallback_metrics(
                DOCUMENT_TEXT, summary, DOCUMENT_CHUNKS[:5]
            )

        return {
            "ok": True,
            "summary": summary,
            "contexts": DOCUMENT_CHUNKS[:5],
            "metrics": {},
            "summarization_metrics": summarization_metrics,
        }
    except Exception as e:
        logger.exception("Summarize failed")
        return {"ok": False, "message": str(e)}

# Generate groundtruth endpoint - UPDATED to handle file uploads
@app.post("/generate_ground_truth")
async def generate_ground_truth_endpoint(
    file: Optional[UploadFile] = File(None),
    doc_id: Optional[str] = Form(None),
    num_questions: int = Form(5),
    question_type: str = Form("mixed"),
    custom_prompt: Optional[str] = Form(None)
):
    """
    Generate ground truth Q&A pairs using DeepEval.
    Accepts either a file upload OR a doc_id.
    """
    try:
        import tempfile
        from ground_truth_data_generation import generate_ground_truth
        
        # Determine the PDF path
        pdf_path = None
        temp_file = None
        
        if file:
            # Handle file upload
            logger.info(f"📝 Generating ground truth from uploaded file: {file.filename}")
            raw = await file.read()
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(raw)
            temp_file.close()
            pdf_path = temp_file.name
            
        elif doc_id:
            # Handle doc_id reference
            logger.info(f"📝 Generating ground truth for doc_id: {doc_id}")
            
            # Check in uploads directory
            uploads_dir = "uploads"
            if os.path.isfile(doc_id):
                pdf_path = doc_id
            elif os.path.isdir(uploads_dir):
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
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'file' or 'doc_id' must be provided"
            )
        
        if not pdf_path or not os.path.isfile(pdf_path):
            raise HTTPException(
                status_code=404,
                detail=f"Could not find PDF for doc_id: {doc_id if doc_id else 'uploaded file'}"
            )
        
        logger.info(f"✓ Using PDF: {pdf_path}")
        logger.info(f"   Questions: {num_questions}, Type: {question_type}")
        
        # Generate ground truth data
        result = await generate_ground_truth(
            doc_id=pdf_path,
            num_questions=num_questions,
            question_type=question_type,
            custom_prompt=custom_prompt
        )
        
        # Clean up temp file if created
        if temp_file:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass
        
        return {
            "ok": True,
            "message": f"Successfully generated {result['count']} ground truth samples",
            "ground_truth_data": result['ground_truth_data'],
            "count": result['count'],
            "file_path": result.get('file_path'),
            "doc_id": doc_id or file.filename if file else "unknown"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ground truth generation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Ground truth generation failed: {str(e)}"
        )

# Add an alias without underscore to avoid 404 from old frontend calls
app.add_api_route(
    "/generate_groundtruth",
    generate_ground_truth_endpoint,
    methods=["POST"],
)

# Download groundtruth endpoint
@app.get("/download_groundtruth")
async def download_groundtruth(file: str):
    """Download the generated groundtruth JSON file."""
    try:
        logger.info(f"Download request for: {file}")
        
        # Validate file path (prevent directory traversal)
        file = os.path.normpath(file)
        if ".." in file or not file.endswith(".goldens.json"):
            logger.warning(f"Invalid file path: {file}")
            return JSONResponse({"ok": False, "message": "Invalid file"}, status_code=400)
        
        if not os.path.exists(file):
            logger.warning(f"File not found: {file}")
            return JSONResponse({"ok": False, "message": "File not found"}, status_code=404)
        
        logger.info(f"✓ Returning file: {file}")
        return FileResponse(file, filename=os.path.basename(file), media_type="application/json")
    except Exception as e:
        logger.exception("Download failed")
        return JSONResponse({"ok": False, "message": str(e)}, status_code=500)

# Evaluate RAG endpoint
@app.post("/evaluate_rag")
def evaluate_rag(
    question: str = Form(...),
    answer: str = Form(...),
    contexts: str = Form(...),
    ground_truth: Optional[str] = Form(None)
):
    """Evaluate a RAG system output against multiple metrics."""
    if eval_engine is None:
        return JSONResponse({"ok": False, "message": "Evaluation engine not initialized"}, status_code=503)
    
    try:
        context_list = json.loads(contexts) if isinstance(contexts, str) else contexts
        
        eval_input = RAGEvalInput(
            question=question,
            answer=answer,
            contexts=context_list,
            ground_truth=ground_truth
        )
        
        result = eval_engine.evaluate(eval_input)
        
        return JSONResponse({
            "ok": True,
            "metrics": eval_engine.get_metrics_summary(result),
            "details": result.metrics_dict
        })
    except Exception as e:
        logger.exception("RAG evaluation failed")
        return JSONResponse(
            {"ok": False, "message": str(e)},
            status_code=500
        )

@app.get("/config")
def get_config():
    """Endpoint to view current LLM configuration"""
    return {
        "llm": {
            "provider": llm_cfg.provider.value,        # fixed var name
            "model": llm_cfg.model,                    # fixed var name
            "temperature": llm_cfg.temperature,        # fixed var name
            "max_tokens": llm_cfg.max_tokens           # fixed var name
        },
        "embedding": {
            "provider": emb_cfg.provider.value,        # fixed var name
            "model": emb_cfg.model,                    # fixed var name
            "dimension": emb_cfg.dimension             # fixed var name
        }
    }

if __name__ == "__main__":
    logger.info("Starting uvicorn server...")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
