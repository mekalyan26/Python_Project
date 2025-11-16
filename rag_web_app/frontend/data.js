const API_BASE = "http://localhost:8000";

const $ = (id) => document.getElementById(id);

let uploadedDocuments = [];
let generatedData = null;

document.addEventListener("DOMContentLoaded", () => {
  console.log("🚀 Data page initialized");
  bindEvents();
  updateStats();
});

function bindEvents() {
  const generateBtn = $("generateBtn");
  const exportBtn = $("exportBtn");
  
  if (generateBtn) {
    generateBtn.addEventListener("click", onGenerateGroundTruth);
  }
  
  if (exportBtn) {
    exportBtn.addEventListener("click", onExportData);
  }
}

async function onGenerateGroundTruth() {
  const gtFile = $("gtFile");
  const numQuestions = $("numQuestions");
  const questionType = $("questionType");
  const customPrompt = $("customPrompt");
  const status = $("generateStatus");
  const preview = $("dataPreview");
  const exportBtn = $("exportBtn");
  
  const file = gtFile?.files?.[0];
  if (!file) {
    setStatus(status, "Please select a PDF file first.", "error");
    return;
  }
  
  try {
    toggleBtn("generateBtn", true, "Generating...");
    setStatus(status, "Uploading file and generating ground truth using DeepEval...", "info");
    
    const formData = new FormData();
    formData.append("file", file);
    formData.append("num_questions", numQuestions?.value || "5");
    formData.append("question_type", questionType?.value || "mixed");
    if (customPrompt?.value?.trim()) {
      formData.append("custom_prompt", customPrompt.value.trim());
    }
    
    console.log("📤 Uploading file:", file.name);
    console.log("   Questions:", numQuestions?.value);
    console.log("   Type:", questionType?.value);
    
    const resp = await fetch(`${API_BASE}/generate_ground_truth`, {
      method: "POST",
      body: formData,
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
      throw new Error(errorMsg);
    }
    
    const data = await resp.json();
    console.log("📥 Full Response:", data);
    
    if (data.ok === false) {
      throw new Error(data.message || "Generation failed");
    }
    
    generatedData = data.ground_truth_data || data.data || [];
    
    console.log("✅ Generated Data:", generatedData);
    console.log(`✅ Count: ${generatedData.length} items`);
    
    if (!Array.isArray(generatedData) || generatedData.length === 0) {
      console.warn("⚠️ No ground truth data in response:", data);
      throw new Error("No ground truth data returned from server");
    }
    
    displayPreview(generatedData);
    
    if (exportBtn) exportBtn.disabled = false;
    
    setStatus(
      status,
      `✅ Successfully generated ${generatedData.length} ground truth samples!`,
      "success"
    );
    
    updateStats();
    
  } catch (err) {
    console.error("❌ Generation error:", err);
    setStatus(status, `Error: ${err.message}`, "error");
    
    if (preview) {
      preview.innerHTML = `
        <div style="padding: 2rem; text-align: center; color: #dc2626;">
          <p style="font-size: 16px; font-weight: 600;">❌ Generation Failed</p>
          <p style="margin-top: 8px; font-size: 14px;">${err.message}</p>
          <p style="margin-top: 12px; font-size: 12px; color: #6b7280;">
            Check console (F12) for details
          </p>
        </div>
      `;
    }
  } finally {
    toggleBtn("generateBtn", false);
  }
}

function displayPreview(data) {
  const preview = $("dataPreview");
  if (!preview) {
    console.warn("⚠️ dataPreview element not found");
    return;
  }

  if (!Array.isArray(data) || data.length === 0) {
    preview.innerHTML = `<div class="gt-empty">No data to display</div>`;
    return;
  }

  console.log(`📊 Displaying ${data.length} samples`);

  const html = `
    <div class="gt-preview">
      ${data.map((item, idx) => `
        <article class="gt-card">
          <div class="gt-title">Sample ${idx + 1}</div>

          <div class="gt-field">
            <div class="gt-label">Question:</div>
            <div class="gt-value">${escapeHtml(item.input || item.question || "N/A")}</div>
          </div>

          <div class="gt-field">
            <div class="gt-label">Expected Answer:</div>
            <div class="gt-value gt-value-expected">${escapeHtml(item.expected_output || item.expected_answer || "N/A")}</div>
          </div>

          ${
            item.actual_output &&
            item.actual_output !== (item.expected_output || item.expected_answer)
              ? `
          <div class="gt-field">
            <div class="gt-label">Actual Output:</div>
            <div class="gt-value gt-value-actual">${escapeHtml(item.actual_output)}</div>
          </div>`
              : ""
          }

          ${
            item.context && Array.isArray(item.context) && item.context.length
              ? `
          <div class="gt-field">
            <div class="gt-label">Context:</div>
            <div class="gt-context">
              ${escapeHtml(item.context[0]).slice(0, 200)}${item.context[0].length > 200 ? "..." : ""}
            </div>
          </div>`
              : ""
          }
        </article>
      `).join("")}
    </div>
  `;

  preview.innerHTML = html;
  
  console.log("✅ Preview rendered successfully");
}

function onExportData() {
  if (!generatedData || generatedData.length === 0) {
    alert("No data to export. Generate ground truth first.");
    return;
  }
  
  try {
    const dataStr = JSON.stringify(generatedData, null, 2);
    const blob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement("a");
    a.href = url;
    a.download = `ground_truth_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    console.log("✅ Data exported successfully");
  } catch (e) {
    console.error("Export error:", e);
    alert(`Export failed: ${e.message}`);
  }
}

function updateStats() {
  const statDocs = $("statDocs");
  const statQuestions = $("statQuestions");
  const statAvg = $("statAvg");
  
  console.log("📊 Updating stats...");
  console.log("   Generated questions:", generatedData?.length || 0);
  
  if (statDocs) statDocs.textContent = "1";
  if (statQuestions) statQuestions.textContent = generatedData?.length || "0";
  if (statAvg) {
    const avg = generatedData?.length || 0;
    statAvg.textContent = avg.toString();
  }
}

function toggleBtn(id, loading, label = "Please wait...") {
  const btn = $(id);
  if (!btn) return;
  
  btn.disabled = !!loading;
  if (loading) {
    btn.dataset._origText = btn.textContent;
    btn.textContent = label;
  } else if (btn.dataset._origText) {
    btn.textContent = btn.dataset._origText;
    delete btn.dataset._origText;
  }
}

function setStatus(el, msg, type = "info") {
  if (!el) return;
  el.textContent = msg;
  el.style.color = 
    type === "success" ? "#065f46" :
    type === "error" ? "#991b1b" :
    "#1e40af";
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = String(text ?? "");
  return div.innerHTML;
}

console.log("✅ data.js loaded");