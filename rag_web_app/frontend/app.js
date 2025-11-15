const API_BASE = "http://localhost:8000";

//Kalyan
const uploadBtn = document.getElementById("uploadBtn");
const pdfFile = document.getElementById("pdfFile");
const uploadStatus = document.getElementById("uploadStatus");

const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("questionInput");
const referenceInput = document.getElementById("referenceInput");
const chatBox = document.getElementById("chatBox");

const summarizeBtn = document.getElementById("summarizeBtn");

// Reference file elements
const referenceFile = document.getElementById("referenceFile");
const referenceFileName = document.getElementById("referenceFileName");

function addMessage(sender, text) {
  const bubble = document.createElement("div");
  const classes = ["p-2", "my-2", "rounded-lg", "max-w-[80%]"];
  if (sender === "user") classes.push("bg-blue-500", "text-white", "self-end", "ml-auto");
  else classes.push("bg-gray-200", "text-gray-800");
  classes.forEach(c => bubble.classList.add(c));
  bubble.innerText = text;
  chatBox.appendChild(bubble);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function showMetrics(metrics) {
  console.log("showMetrics (fallback) called with:", metrics);
  const set = (id, v) => {
    const el = document.getElementById(id);
    if (!el) {
      console.warn("Missing metric element:", id);
      return;
    }
    const value = (typeof v === "number") ? v.toFixed(4) : (v ?? "0.0000");
    el.textContent = value;
    console.log(`  Set ${id} = ${value}`);
  };

  const m = metrics || {};
  console.log("Fallback metrics object:", m);
  set("m_answer_relevancy", m.answer_relevancy ?? m.relevancy ?? 0.0);
  set("m_faithfulness", m.faithfulness ?? 0.0);
  set("m_contextual_recall", m.contextual_recall ?? m.recall ?? 0.0);
  set("m_contextual_precision", m.contextual_precision ?? m.precision ?? 0.0);
  set("m_contextual_relevancy", m.contextual_relevancy ?? 0.0);
  set("m_ragas", m.ragas ?? m.ragas_score ?? 0.0);

  const panel = document.getElementById("metricsPanel");
  if (panel) {
    panel.style.display = "block";
    console.log("✓ Fallback metrics panel displayed");
  }
}

function showDeepEvalMetrics(deepevalMetrics) {
  console.log("===== showDeepEvalMetrics called =====");
  console.log("Raw deepevalMetrics object:", deepevalMetrics);
  console.log("Type:", typeof deepevalMetrics);
  console.log("Is empty object?", Object.keys(deepevalMetrics || {}).length === 0);

  if (!deepevalMetrics || Object.keys(deepevalMetrics).length === 0) {
    console.warn("⚠️ No DeepEval metrics received - using empty defaults");
  }

  const set = (id, v) => {
    const el = document.getElementById(id);
    if (!el) {
      console.error(`❌ Missing DeepEval metric element: ${id}`);
      return;
    }
    const value = (typeof v === "number") ? v.toFixed(4) : (v ?? "0.0000");
    el.textContent = value;
    console.log(`  ✓ Set ${id} = ${value}`);
  };

  const d = deepevalMetrics || {};
  console.log("DeepEval metrics keys:", Object.keys(d));
  
  set("d_answer_relevancy", d.answer_relevancy ?? d.answer_relevancy_score ?? 0.0);
  set("d_faithfulness", d.faithfulness ?? d.faithfulness_score ?? 0.0);
  set("d_contextual_recall", d.contextual_recall ?? d.contextual_recall_score ?? 0.0);
  set("d_contextual_precision", d.contextual_precision ?? d.contextual_precision_score ?? 0.0);
  set("d_ragas", d.ragas ?? d.ragas_score ?? 0.0);

  const panel = document.getElementById("deepevalMetricsPanel");
  if (panel) {
    panel.style.display = "block";
    console.log("✓ DeepEval metrics panel displayed");
  } else {
    console.error("❌ deepevalMetricsPanel not found in DOM");
  }
  console.log("===== showDeepEvalMetrics complete =====\n");
}

function showSummarizationMetrics(s) {
  const set = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.textContent = (typeof v === "number") ? v.toFixed(4) : (v ?? "0.0000");
  };
  if (!s || Object.keys(s).length === 0) {
    const panel = document.getElementById("summarizationMetricsPanel");
    if (panel) panel.style.display = "none";
    return;
  }
  set("s_summarization", s.summarization ?? 0.0);
  set("s_hallucination", s.hallucination ?? 0.0);
  set("s_bias", s.bias ?? 0.0);
  set("s_toxicity", s.toxicity ?? 0.0);
  set("s_readability", s.readability ?? 0.0);
  const panel = document.getElementById("summarizationMetricsPanel");
  if (panel) panel.style.display = "block";
}

function showWaitingSpinner() {
  if (document.getElementById("awaitSpinner")) return;
  const wrapper = document.createElement("div");
  wrapper.id = "awaitSpinner";
  wrapper.className = "waiting-bubble";
  const spinner = document.createElement("div");
  spinner.className = "spinner";
  for (let i = 0; i < 12; i++) {
    const blade = document.createElement("div");
    blade.className = "spinner-blade";
    spinner.appendChild(blade);
  }
  wrapper.appendChild(spinner);
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function removeWaitingSpinner() {
  const el = document.getElementById("awaitSpinner");
  if (el) el.remove();
}

uploadBtn?.addEventListener("click", async () => {
  const file = pdfFile?.files?.[0];
  if (!file) {
    uploadStatus.textContent = "Please select a PDF file.";
    return;
  }
  uploadStatus.textContent = "Uploading and indexing...";
  const formData = new FormData();
  formData.append("file", file);
  formData.append("chunk_size", "900");
  formData.append("chunk_overlap", "200");
  try {
    const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
    const data = await res.json();
    uploadStatus.textContent = data.message || "Upload completed.";
  } catch (err) {
    console.error("Upload failed:", err);
    uploadStatus.textContent = "Upload failed: " + err.message;
  }
});

askBtn?.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  const reference = referenceInput?.value.trim() || null;
  if (!question) return;
  
  console.log("\n========== ASK REQUEST ==========");
  console.log("Question:", question);
  console.log("Reference answer provided:", !!reference);
  
  addMessage("user", question);
  questionInput.value = "";

  try {
    showWaitingSpinner();

    const payload = { question };
    if (reference) payload.reference_answer = reference;

    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    removeWaitingSpinner();

    const data = await res.json();
    console.log("\n========== /ask RESPONSE ==========");
    console.log("Full response:", data);
    console.log("response.metrics:", data.metrics);
    console.log("response.deepeval_metrics:", data.deepeval_metrics);
    console.log("response.fallback_eval_metrics:", data.fallback_eval_metrics);

    if (!res.ok) {
      addMessage("bot", `⚠️ ${data?.message || res.statusText}`);
      console.error("Response error:", res.status, data.message);
      showMetrics({});
      showDeepEvalMetrics({});
      return;
    }

    addMessage("bot", data.answer ?? JSON.stringify(data));
    
    // Show fallback metrics from RAG pipeline
    console.log("\n--- Displaying Fallback Metrics (from RAG pipeline) ---");
    showMetrics(data.metrics ?? {});
    
    // Show DeepEval metrics (prioritize fallback_eval_metrics, then deepeval_metrics)
    console.log("\n--- Displaying DeepEval Metrics ---");
    const deepevalToShow = data.deepeval_metrics || data.fallback_eval_metrics || {};
    console.log("DeepEval metrics to display:", deepevalToShow);
    showDeepEvalMetrics(deepevalToShow);
    
    console.log("========== ASK REQUEST COMPLETE ==========\n");
  } catch (err) {
    removeWaitingSpinner();
    console.error("Request error:", err);
    addMessage("bot", `❌ Request failed: ${err.message}`);
  }
});

summarizeBtn?.addEventListener("click", async () => {
  try {
    showWaitingSpinner();
    const res = await fetch(`${API_BASE}/summarize`);
    const data = await res.json();
    removeWaitingSpinner();

    if (!res.ok || !data.ok) {
      addMessage("bot", `⚠️ ${data?.message || res.statusText}`);
      showSummarizationMetrics({});
      return;
    }

    addMessage("bot", data.summary || "No summary returned");
    showSummarizationMetrics(data.summarization_metrics || {});
  } catch (err) {
    removeWaitingSpinner();
    addMessage("bot", `❌ Error: ${err.message}`);
  }
});

// Apply retro styles to existing controls at runtime
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("button").forEach(b => b.classList.add("retro-btn"));
  document.getElementById("askBtn")?.classList.add("retro-btn--accent");
  document.getElementById("uploadBtn")?.classList.add("retro-btn");
  document.getElementById("summarizeBtn")?.classList.add("retro-btn");

  document.querySelectorAll('input[type="text"], input[type="search"]').forEach(i => i.classList.add("retro-input"));
});

// Reference file -> auto-fill referenceInput
referenceFile?.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  referenceFileName.textContent = file.name;

  try {
    const text = await file.text();
    if (referenceInput) {
      // Limit to avoid huge payloads
      referenceInput.value = text.slice(0, 8000);
    }
    console.log("Loaded reference file:", { name: file.name, size: file.size });
  } catch (err) {
    console.error("Failed to read reference file:", err);
  }
});