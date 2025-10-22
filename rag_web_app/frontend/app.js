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
  bubble.classList.add(
    "p-2",
    "my-2",
    "rounded-lg",
    "max-w-[80%]",
    sender === "user"
      ? "bg-blue-500 text-white self-end ml-auto"
      : "bg-gray-200 text-gray-800"
  );
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
    if (data.ok) {
      uploadStatus.textContent = `✅ Indexed ${data.chunks} chunks successfully.`;
    } else {
      uploadStatus.textContent = `❌ Error: ${data.message}`;
    }
  } catch (err) {
    console.error(err);
    uploadStatus.textContent = "❌ Upload failed.";
  }
});

askBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";

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
});

summarizeBtn.addEventListener("click", async () => {
  addMessage("user", "Summarize this document.");
  const res = await fetch(`${API_BASE}/summarize`);
  const data = await res.json();
  if (data.ok) {
    addMessage("bot", data.summary);
  } else {
    addMessage("bot", `⚠️ ${data.message}`);
  }
});
