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
  console.log("showMetrics called with:", metrics);
  const set = (id, v) => {
    const el = document.getElementById(id);
    if (!el) return console.warn("Missing metric element:", id);
    el.textContent = (typeof v === "number") ? v.toFixed(4) : (v ?? "0.0000");
  };
  const m = metrics || {};
  set("m_answer_relevancy", m.answer_relevancy ?? m.relevancy ?? 0.0);
  set("m_faithfulness", m.faithfulness ?? 0.0);
  set("m_contextual_recall", m.contextual_recall ?? m.recall ?? 0.0);
  set("m_contextual_precision", m.contextual_precision ?? m.precision ?? 0.0);
  set("m_contextual_relevancy", m.contextual_relevancy ?? 0.0);
  set("m_ragas", m.ragas ?? m.ragas_score ?? 0.0);

  const panel = document.getElementById("metricsPanel");
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

    // parse JSON and handle errors
    const data = await res.json();
    console.log("/ask response:", data);

    removeWaitingSpinner();

    if (!res.ok) {
      addMessage("bot", `⚠️ ${data?.message || res.statusText}`);
      showMetrics({});
      return;
    }

    addMessage("bot", data.answer ?? "No answer returned");
    showMetrics(data.metrics ?? {});
  } catch (err) {
    removeWaitingSpinner();
    console.error("Request error:", err);
    addMessage("bot", `❌ Request failed: ${err.message}`);
    showMetrics({});
  }
});

summarizeBtn?.addEventListener("click", async () => {
  try {
    showWaitingSpinner();
    const res = await fetch(`${API_BASE}/summarize`);
    const data = await res.json();
    console.log("/summarize response:", data);
    removeWaitingSpinner();
    if (!res.ok) {
      addMessage("bot", `⚠️ ${data?.message || res.statusText}`);
      showMetrics({});
      return;
    }
    addMessage("bot", data.summary ?? "No summary returned");
    showMetrics(data.metrics ?? {});
  } catch (err) {
    removeWaitingSpinner();
    console.error("Summarize error:", err);
    addMessage("bot", `❌ Error: ${err.message}`);
    showMetrics({});
  }
});