const API_BASE = "http://localhost:8000";

const uploadBtn = document.getElementById("uploadBtn");
const pdfFile = document.getElementById("pdfFile");
const uploadStatus = document.getElementById("uploadStatus");

const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("questionInput");
const chatBox = document.getElementById("chatBox");

const summarizeBtn = document.getElementById("summarizeBtn");
const referenceInput = document.getElementById("referenceInput");

function addMessage(sender, text) {
  const bubble = document.createElement("div");
  const classes = [
    "p-2",
    "my-2",
    "rounded-lg",
    "max-w-[80%]"
  ];
  
  if (sender === "user") {
    classes.push("bg-blue-500", "text-white", "self-end", "ml-auto");
  } else {
    classes.push("bg-gray-200", "text-gray-800");
  }
  
  classes.forEach(cls => bubble.classList.add(cls));
  bubble.innerText = text;
  chatBox.appendChild(bubble);
  chatBox.scrollTop = chatBox.scrollHeight;
}

uploadBtn.addEventListener("click", async () => {
  const file = pdfFile.files[0];
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
    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    uploadStatus.textContent = data.message;
  } catch (err) {
    console.error("Upload failed:", err);
    uploadStatus.textContent = "Upload failed: " + err.message;
  }
});

askBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  const reference = referenceInput.value.trim() || null;
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

    removeWaitingSpinner();

    const text = await res.text();
    let data;
    try {
      data = text ? JSON.parse(text) : null;
    } catch (parseErr) {
      addMessage("bot", `❌ Invalid response from server`);
      console.error("parse error", parseErr, text);
      return;
    }

    if (!res.ok) {
      addMessage("bot", `⚠️ ${data?.message || res.statusText}`);
      return;
    }

    // show answer
    addMessage("bot", data.answer ?? "No answer returned");

    // show metrics panel
    showMetrics(data.metrics ?? {});
  } catch (err) {
    removeWaitingSpinner();
    console.error("Request error:", err);
    addMessage("bot", `❌ Request failed: ${err.message}`);
  }
});

summarizeBtn.addEventListener("click", async () => {
  addMessage("user", "Summarize this document.");
  try {
    const res = await fetch(`${API_BASE}/summarize`);
    const data = await res.json();
    if (data.ok) {
      addMessage("bot", data.summary);
    } else {
      addMessage("bot", `⚠️ ${data.message}`);
    }
  } catch (err) {
    addMessage("bot", `❌ Error: ${err.message}`);
  }
});

function showMetrics(metrics) {
  const metricsPanel = document.getElementById("metricsPanel");
  
  // For debugging - log the metrics
  console.log('Received metrics:', metrics);

  // Set default values and update display
  const defaults = {
    answer_relevancy: "0.0",
    faithfulness: "0.0",
    contextual_recall: "0.0",
    contextual_precision: "0.0",
    contextual_relevancy: "0.0",
    ragas: "0.0"
  };

  // Update each metric element
  Object.keys(defaults).forEach(key => {
    const elementId = `m_${key}`;
    const element = document.getElementById(elementId);
    if (element) {
      element.textContent = metrics?.[key] ?? defaults[key];
    }
  });

  // Show the panel
  metricsPanel.style.display = 'block';
}

// Assuming you have a function to handle the response
function handleResponse(response) {
    const answer = response.answer;
    const metrics = response.metrics;

    // Display the answer
    document.getElementById("answer").innerText = answer;

    // Display the metrics
    document.getElementById("relevancy").innerText = metrics.answer_relevancy || "N/A";
    document.getElementById("faithfulness").innerText = metrics.faithfulness || "N/A";
    document.getElementById("recall").innerText = metrics.contextual_recall || "N/A";
    document.getElementById("precision").innerText = metrics.contextual_precision || "N/A";
    document.getElementById("contextual-relevancy").innerText = metrics.contextual_relevancy || "N/A";
    document.getElementById("ragas").innerText = metrics.ragas || "N/A";
}

function showWaitingSpinner() {
  document.getElementById("waitingSpinner").classList.remove("hidden");
}

function removeWaitingSpinner() {
  document.getElementById("waitingSpinner").classList.add("hidden");
}