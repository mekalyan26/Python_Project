const API_BASE = "http://localhost:8000";

const uploadBtn = document.getElementById("uploadBtn");
const pdfFile = document.getElementById("pdfFile");
const uploadStatus = document.getElementById("uploadStatus");

const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("questionInput");
const chatBox = document.getElementById("chatBox");

const summarizeBtn = document.getElementById("summarizeBtn");

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
  if (!question) return;  // Add return statement here

  addMessage("user", question);
  questionInput.value = "";

  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();
    if (data.ok) {
      addMessage("bot", data.answer);
    } else {
      addMessage("bot", `⚠️ ${data.message}`);
    }
  } catch (err) {
    console.error("Request failed:", err);
    addMessage("bot", `❌ Error: ${err.message}`);
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