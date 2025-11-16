import os, re, heapq, logging
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import shutil
import json

# Load environment variables from .env file
load_dotenv()  # This loads .env file from current directory

# Verify API key is loaded
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print(f"✓ OPENAI_API_KEY loaded (length: {len(openai_key)})")
else:
    print("⚠️  OPENAI_API_KEY not found in environment")

# Configure logging first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
            top_k=req.top_k or 4,
            max_new_tokens=req.max_new_tokens or 512,
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
                        # use internal deepeval evaluation method if present
                        if hasattr(eval_engine, "_deepeval_evaluate"):
                            de_result = eval_engine._deepeval_evaluate(eval_input)
                            deepeval_metrics = eval_engine.get_metrics_summary(de_result)
                        else:
                            # fall back to public evaluate (may choose deepeval or fallback internally)
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

        return {
            "ok": True,
            "summary": summary,
            "contexts": DOCUMENT_CHUNKS[:5],
            "metrics": {},  # keep existing shape
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

if __name__ == "__main__":
    logger.info("Starting uvicorn server...")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
