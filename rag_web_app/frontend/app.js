const API_BASE = "http://localhost:8000";

let state = {
  docId: null,
  indexed: false,
  uploading: false,
};

const $ = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", () => {
  console.log("🚀 RAG App initialized");
  bindEvents();
  if ($("askBtn")) $("askBtn").disabled = true;
});

function bindEvents() {
  const uploadBtn = $("uploadBtn");
  const askBtn = $("askBtn");
  const summarizeBtn = $("summarizeBtn");
  const refFile = $("referenceFile");
  
  if (uploadBtn) {
    uploadBtn.addEventListener("click", onUploadAndIndex);
  }
  
  if (askBtn) {
    askBtn.addEventListener("click", onAsk);
  }
  
  if (summarizeBtn) {
    summarizeBtn.addEventListener("click", onSummarize);
  }
  
  if (refFile) {
    refFile.addEventListener("change", (e) => {
      const name = e.target?.files?.[0]?.name || "";
      const label = $("referenceFileName");
      if (label) label.textContent = name || "Choose .txt / .md / .json";
    });
  }
}

/* --- Upload + Index --- */
async function onUploadAndIndex() {
  if (state.uploading) return;
  
  const status = $("uploadStatus");
  const fileInput = $("pdfFile");
  
  if (!fileInput?.files?.length) {
    setStatus(status, "Please choose a PDF file.", "error");
    return;
  }

  const file = fileInput.files[0];
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setStatus(status, "Only PDF files are supported.", "error");
    return;
  }

  try {
    state.uploading = true;
    toggleBtn("uploadBtn", true, "Uploading…");
    setStatus(status, "Uploading PDF…", "info");

    const form = new FormData();
    form.append("file", file);

    const uploadResp = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: form,
    });

    if (!uploadResp.ok) {
      const errorText = await uploadResp.text();
      throw new Error(`Upload failed: ${errorText}`);
    }

    const uploadJson = await uploadResp.json();
    console.log("📥 Upload response:", uploadJson);

    state.docId = uploadJson.doc_id || 
                  uploadJson.document_id || 
                  uploadJson.index_id || 
                  uploadJson.id || 
                  uploadJson.file_id ||
                  null;

    state.indexed = true;
    const askBtn = $("askBtn");
    if (askBtn) askBtn.disabled = false;
    
    setStatus(status, uploadJson.message || "✓ Upload and indexing complete.", "success");
    console.log("✅ Document indexed successfully");
  } catch (err) {
    console.error("❌ Upload/Index error:", err);
    setStatus(status, `Error: ${err.message}`, "error");
  } finally {
    state.uploading = false;
    toggleBtn("uploadBtn", false);
  }
}

/* --- Ask --- */
async function onAsk() {
  const questionInput = $("questionInput");
  if (!questionInput) return;
  
  const question = questionInput.value?.trim();
  if (!question) {
    toast("Please enter a question.", "error");
    return;
  }
  
  if (!state.indexed && !state.docId) {
    toast("Please upload and index a PDF first.", "error");
    return;
  }

  try {
    toggleBtn("askBtn", true, "Asking…");
    appendMessage("You", question);
    console.log(`❓ Asking: ${question}`);

    const reference_answer = await getReferenceAnswer();
    
    const payload = {
      question: question,
      doc_id: state.docId || null,
      top_k: 5,
      k: 5,
      num_results: 5,
      reference_answer: reference_answer || null,
      expected_answer: reference_answer || null,
      ground_truth: reference_answer || null,
      evaluate: !!reference_answer,
      use_reranker: false,
      stream: false,
      temperature: 0.7,
      max_tokens: 500,
    };

    console.log("📤 Request payload:", payload);

    const resp = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const errorText = await resp.text();
      let errorMsg;
      try {
        const errorJson = JSON.parse(errorText);
        errorMsg = errorJson.detail || errorJson.message || errorText;
      } catch {
        errorMsg = errorText;
      }
      throw new Error(`Ask failed: ${errorMsg}`);
    }
    
    const data = await resp.json();
    console.log("📥 Ask response:", data);

    const answer = data.answer || data.response || data.result || "No answer returned.";
    appendMessage("Assistant", answer);
    
    if (Array.isArray(data.sources) && data.sources.length) {
      appendMessage("Sources", data.sources.map(s => s.title || s.id || s).join("; "));
    }

    if (data.metrics) {
      console.log("📊 Updating DeepEval metrics:", data.metrics);
      updateDeepEvalMetrics(data.metrics);
    }
    if (data.fallback_metrics) {
      console.log("📊 Updating fallback metrics:", data.fallback_metrics);
      updateFallbackMetrics(data.fallback_metrics);
    }
    
    console.log("✅ Question answered successfully");
  } catch (err) {
    console.error("❌ Ask error:", err);
    appendMessage("System", `Error: ${err.message}`);
    toast(err.message, "error");
  } finally {
    toggleBtn("askBtn", false);
  }
}

/* --- Summarize with GET/POST fallback --- */
async function onSummarize() {
  if (!state.indexed && !state.docId) {
    toast("Please upload and index a PDF first.", "error");
    return;
  }
  
  try {
    toggleBtn("summarizeBtn", true, "Summarizing…");
    console.log("📝 Requesting summary...");
    
    let resp;
    let success = false;
    
    // Try POST first (most common for REST APIs)
    try {
      console.log("🔍 Trying POST /summarize");
      const payload = {
        doc_id: state.docId || null,
        document_id: state.docId || null,
        max_length: 500,
        min_length: 100,
      };

      resp = await fetch(`${API_BASE}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (resp.ok) {
        success = true;
        console.log("✅ POST /summarize succeeded");
      } else {
        console.warn(`⚠ POST /summarize failed: ${resp.status}`);
      }
    } catch (e) {
      console.warn("⚠ POST /summarize error:", e.message);
    }

    // If POST failed, try GET with query params
    if (!success) {
      try {
        console.log("🔍 Trying GET /summarize with query params");
        const params = new URLSearchParams({
          doc_id: state.docId || "",
          max_length: "500",
          min_length: "100",
        });

        resp = await fetch(`${API_BASE}/summarize?${params}`, {
          method: "GET",
        });

        if (resp.ok) {
          success = true;
          console.log("✅ GET /summarize succeeded");
        } else {
          console.warn(`⚠ GET /summarize failed: ${resp.status}`);
        }
      } catch (e) {
        console.warn("⚠ GET /summarize error:", e.message);
      }
    }

    // Try alternative endpoints
    if (!success) {
      const alternatives = [
        { path: "/summary", method: "POST" },
        { path: "/api/summarize", method: "POST" },
        { path: "/documents/summarize", method: "POST" },
        { path: `/summarize/${state.docId}`, method: "GET" },
      ];

      for (const alt of alternatives) {
        try {
          console.log(`🔍 Trying ${alt.method} ${alt.path}`);
          
          if (alt.method === "POST") {
            resp = await fetch(`${API_BASE}${alt.path}`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ doc_id: state.docId }),
            });
          } else {
            resp = await fetch(`${API_BASE}${alt.path}`, {
              method: "GET",
            });
          }

          if (resp.ok) {
            success = true;
            console.log(`✅ ${alt.method} ${alt.path} succeeded`);
            break;
          }
        } catch (e) {
          console.warn(`⚠ ${alt.method} ${alt.path} error:`, e.message);
        }
      }
    }

    if (!success) {
      throw new Error(
        "Summarize endpoint not found or not configured correctly.\n\n" +
        "Tried:\n" +
        "  • POST /summarize\n" +
        "  • GET /summarize?doc_id=...\n" +
        "  • POST /summary\n" +
        "  • POST /api/summarize\n\n" +
        "Please check your backend's /docs for the correct endpoint."
      );
    }
    
    const data = await resp.json();
    console.log("📥 Summary response:", data);

    const summary = data.summary || data.text || data.result || "No summary returned.";
    appendMessage("Assistant", `📝 Summary:\n\n${summary}`);

    // FIX: pick correct metrics object
    const summaryMetricsObj =
      data.summarization_metrics ||                // preferred key
      data.metrics?.summarization_metrics ||       // nested variant
      data.metrics ||                              // legacy structure
      null;

    console.log("🔎 Extracted summarization metrics:", summaryMetricsObj);

    if (summaryMetricsObj) {
      updateSummaryMetrics(summaryMetricsObj);
      const panel = $("summarizationMetricsPanel");
      if (panel) panel.style.display = "block";
    } else {
      console.warn("⚠ No summarization metrics found in response.");
    }

    if (data.metrics) {
      // Keep DeepEval style metrics separate
      updateDeepEvalMetrics(data.metrics);
    }
    if (data.fallback_metrics) {
      updateFallbackMetrics(data.fallback_metrics);
    }

    console.log("✅ Summary generated successfully");
  } catch (err) {
    console.error("❌ Summarize error:", err);
    appendMessage("System", `❌ ${err.message}`);
    toast(err.message, "error");
  } finally {
    toggleBtn("summarizeBtn", false);
  }
}

/* --- UI Helpers --- */
function appendMessage(author, text) {
  const chat = $("chatBox");
  if (!chat) return;
  
  const who = document.createElement("div");
  who.style.fontWeight = "600";
  who.style.fontSize = "14px";
  who.style.color = author === "You" ? "#374151" : author === "System" ? "#dc2626" : "#2563eb";
  who.style.marginBottom = "4px";
  who.textContent = author;
  
  const msg = document.createElement("div");
  msg.style.marginBottom = "12px";
  msg.style.fontSize = "15px";
  msg.style.whiteSpace = "pre-wrap";
  msg.textContent = text;
  
  chat.appendChild(who);
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

function toggleBtn(id, loading, labelWhenLoading = "Please wait…") {
  const btn = $(id);
  if (!btn) return;
  
  btn.disabled = !!loading;
  if (loading) {
    btn.dataset._label = btn.textContent;
    btn.textContent = labelWhenLoading;
  } else if (btn.dataset._label) {
    btn.textContent = btn.dataset._label;
    delete btn.dataset._label;
  }
}

function setStatus(el, msg, type = "info") {
  if (!el) return;
  el.textContent = msg;
  el.style.color = type === "success" ? "#065f46" : type === "error" ? "#991b1b" : "#1e40af";
}

function toast(msg, type = "info") {
  const s = $("uploadStatus");
  if (s) {
    setStatus(s, msg, type);
  } else {
    console.warn("⚠ uploadStatus element not found");
  }
}

async function getReferenceAnswer() {
  const inline = $("referenceInput")?.value?.trim();
  const fileInput = $("referenceFile");
  
  if (fileInput?.files?.length) {
    try {
      const file = fileInput.files[0];
      const text = await file.text();
      return text?.trim() || inline || "";
    } catch (e) {
      console.warn("Could not read reference file:", e);
      return inline || "";
    }
  }
  return inline || "";
}

/* --- Metrics Updaters --- */
function updateDeepEvalMetrics(m) {
  setNum("d_answer_relevancy", m.answer_relevancy ?? m.answerRelevancy ?? m.answer_relevance);
  setNum("d_faithfulness", m.faithfulness);
  setNum("d_contextual_recall", m.contextual_recall ?? m.contextualRecall ?? m.context_recall);
  setNum("d_contextual_precision", m.contextual_precision ?? m.contextualPrecision ?? m.context_precision);
  setNum("d_ragas", m.ragas ?? m.ragas_score);
}

function updateFallbackMetrics(m) {
  setNum("m_answer_relevancy", m.answer_relevancy ?? m.answerRelevancy ?? m.answer_relevance);
  setNum("m_faithfulness", m.faithfulness);
  setNum("m_contextual_recall", m.contextual_recall ?? m.contextualRecall ?? m.context_recall);
  setNum("m_contextual_precision", m.contextual_precision ?? m.contextualPrecision ?? m.context_precision);
  setNum("m_contextual_relevancy", m.contextual_relevancy ?? m.contextualRelevancy ?? m.context_relevancy);
  setNum("m_ragas", m.ragas ?? m.ragas_score);
}

function updateSummaryMetrics(m) {
  // Added explicit logging for debugging
  console.log("🧮 Updating summary metrics with object:", m);
  setNum("s_summarization", m.summarization ?? m.quality ?? m.score);
  setNum("s_hallucination", m.hallucination);
  setNum("s_bias", m.bias);
  setNum("s_toxicity", m.toxicity);
  setNum("s_readability", m.readability);
}

function setNum(id, v) {
  const el = $(id);
  if (!el) return;
  const n = Number(v);
  if (!Number.isFinite(n)) return;
  el.textContent = n.toFixed(4);
}

console.log("✅ app.js loaded successfully");
