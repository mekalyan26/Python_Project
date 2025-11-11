import os
import tempfile
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_utils import extract_text_from_pdf, chunk_text
from rag_pipeline import RAGEngine

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local RAG Backend", version="0.1.0")

# CORS (adjust for your frontend origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to ["http://localhost:5173"] etc. in real use
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGEngine()  # single-engine in-memory state

class AskRequest(BaseModel):
    question: str
    reference_answer: Optional[str] = None
    top_k: Optional[int] = 4
    max_new_tokens: Optional[int] = 512

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), chunk_size: int = Form(900), chunk_overlap: int = Form(200)):
    # Save temp file
    suffix = ".pdf" if not file.filename.endswith(".pdf") else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Extract + chunk
    text = extract_text_from_pdf(tmp_path)
    os.remove(tmp_path)

    if not text.strip():
        return {"ok": False, "message": "No extractable text found in PDF."}

    rag.full_text = text
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return {"ok": False, "message": "Unable to build chunks from document."}

    rag.build_index(chunks)
    return {
        "ok": True,
        "chunks": len(chunks),
        "message": "PDF processed and indexed.",
    }

@app.post("/ask")
def ask(req: AskRequest):
    if rag is None:
        return {"ok": False, "message": "RAG engine not initialized"}
    
    logger.debug(f"Processing question: {req.question[:100]}...")
    result = rag.answer(
        req.question,
        reference_answer=req.reference_answer,
        top_k=req.top_k or 4,
        max_new_tokens=req.max_new_tokens or 512,
    )
    logger.debug(f"Metrics computed: {result.get('metrics', {})}")
    return {"ok": True, **result}

@app.get("/summarize")
def summarize(max_new_tokens: int = 600):
    if rag.index is None:
        return {"ok": False, "message": "No document indexed. Upload a PDF first."}
    summary = rag.summarize(max_new_tokens=max_new_tokens)
    return {"ok": True, "summary": summary}
