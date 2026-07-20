/**
 * app.js
 * ------
 * Frontend logic for Equation Slate.
 *   - Tab switching between "draw" and "upload" input modes
 *   - Freehand drawing canvas (mouse + touch/stylus)
 *   - File upload with drag-and-drop + preview
 *   - Calls POST /api/recognize and renders the JSON response
 *
 * No build step / framework: this is intentionally plain DOM + fetch so the
 * whole frontend ships as two static files.
 */

(() => {
  "use strict";

  // ------------------------------------------------------------------
  // Elements
  // ------------------------------------------------------------------
  const statusPill = document.getElementById("statusPill");
  const statusText = document.getElementById("statusText");

  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  const canvas = document.getElementById("drawCanvas");
  const ctx = canvas.getContext("2d");
  const clearCanvasBtn = document.getElementById("clearCanvas");
  const penWidthInput = document.getElementById("penWidth");

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const previewWrap = document.getElementById("previewWrap");
  const previewImage = document.getElementById("previewImage");

  const solveBtn = document.getElementById("solveBtn");
  const btnLabel = solveBtn.querySelector(".btn-label");
  const btnSpinner = solveBtn.querySelector(".btn-spinner");
  const errorMsg = document.getElementById("errorMsg");

  const resultsEmpty = document.getElementById("resultsEmpty");
  const resultsBody = document.getElementById("resultsBody");
  const confidenceBadge = document.getElementById("confidenceBadge");
  const confidenceValue = document.getElementById("confidenceValue");
  const latexOutput = document.getElementById("latexOutput");
  const solutionOutput = document.getElementById("solutionOutput");
  const samplesList = document.getElementById("samplesList");
  const copyLatexBtn = document.getElementById("copyLatexBtn");
  const copySolutionBtn = document.getElementById("copySolutionBtn");

  let activeTab = "draw";
  let uploadedFile = null;
  let currentSolutionText = "";

  // ------------------------------------------------------------------
  // Health check
  // ------------------------------------------------------------------
  fetch("/healthz")
    .then((r) => r.json())
    .then((data) => {
      statusPill.classList.add(data.model_loaded ? "ready" : "ready");
      statusText.textContent = data.model_loaded
        ? "Model ready"
        : "Model loads on first request";
    })
    .catch(() => {
      statusPill.classList.add("error");
      statusText.textContent = "Server unreachable";
    });

  // ------------------------------------------------------------------
  // Tabs
  // ------------------------------------------------------------------
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      activeTab = btn.dataset.tab;
      tabButtons.forEach((b) => b.classList.toggle("active", b === btn));
      tabPanels.forEach((p) => p.classList.toggle("active", p.id === `tab-${activeTab}`));
      hideError();
    });
  });

  // ------------------------------------------------------------------
  // Drawing canvas
  // ------------------------------------------------------------------
  let drawing = false;
  let lastPoint = null;

  function initCanvas() {
    canvas.width = 800;
    canvas.height = 360;
    primeCanvas();
  }

  function primeCanvas() {
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#000000";
  }

  function getPoint(evt) {
    const rect = canvas.getBoundingClientRect();
    const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
    const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
      scale: (scaleX + scaleY) / 2
    };
  }

  function startDraw(evt) {
    evt.preventDefault();
    drawing = true;
    lastPoint = getPoint(evt);
  }

  function moveDraw(evt) {
    if (!drawing) return;
    evt.preventDefault();
    const point = getPoint(evt);
    const penVal = parseInt(penWidthInput.value, 10);
    ctx.lineWidth = penVal * point.scale;
    ctx.beginPath();
    ctx.moveTo(lastPoint.x, lastPoint.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    lastPoint = point;
  }

  function endDraw() {
    drawing = false;
    lastPoint = null;
  }

  canvas.addEventListener("mousedown", startDraw);
  canvas.addEventListener("mousemove", moveDraw);
  window.addEventListener("mouseup", endDraw);

  canvas.addEventListener("touchstart", startDraw, { passive: false });
  canvas.addEventListener("touchmove", moveDraw, { passive: false });
  canvas.addEventListener("touchend", endDraw);

  initCanvas();

  clearCanvasBtn.addEventListener("click", () => {
    primeCanvas();
    hideError();
  });

  function canvasHasInk() {
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 0; i < data.length; i += 4 * 37) {
      // sample every ~37th pixel for speed; any non-white pixel counts as ink
      if (data[i] < 250 || data[i + 1] < 250 || data[i + 2] < 250) return true;
    }
    return false;
  }

  // ------------------------------------------------------------------
  // Upload
  // ------------------------------------------------------------------
  dropzone.addEventListener("click", () => fileInput.click());

  ["dragenter", "dragover"].forEach((evtName) => {
    dropzone.addEventListener(evtName, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evtName) => {
    dropzone.addEventListener(evtName, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });

  function handleFile(file) {
    uploadedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewWrap.hidden = false;
    };
    reader.readAsDataURL(file);
    hideError();
  }

  // ------------------------------------------------------------------
  // Submit
  // ------------------------------------------------------------------
  solveBtn.addEventListener("click", async () => {
    hideError();

    let body;
    let headers = {};

    if (activeTab === "draw") {
      if (!canvasHasInk()) {
        showError("The canvas is empty — draw an equation first.");
        return;
      }
      const dataUrl = canvas.toDataURL("image/png");
      body = JSON.stringify({ image_data: dataUrl });
      headers["Content-Type"] = "application/json";
    } else {
      if (!uploadedFile) {
        showError("Choose or drop an image first.");
        return;
      }
      const formData = new FormData();
      formData.append("image", uploadedFile);
      body = formData;
    }

    setLoading(true);
    try {
      const response = await fetch("/api/recognize", {
        method: "POST",
        headers,
        body,
      });
      const data = await response.json();

      if (!response.ok) {
        showError(data.error || "Something went wrong.");
        return;
      }

      renderResults(data);
    } catch (err) {
      showError("Could not reach the server. Please try again.");
    } finally {
      setLoading(false);
    }
  });

  function setLoading(isLoading) {
    solveBtn.disabled = isLoading;
    btnSpinner.hidden = !isLoading;
    btnLabel.textContent = isLoading ? "Reading your handwriting…" : "Recognize & Solve";
  }

  function showError(message) {
    errorMsg.textContent = message;
    errorMsg.hidden = false;
  }

  function hideError() {
    errorMsg.hidden = true;
    errorMsg.textContent = "";
  }

  // ------------------------------------------------------------------
  // Render results
  // ------------------------------------------------------------------
  function renderResults(data) {
    resultsEmpty.hidden = true;
    resultsBody.hidden = false;

    confidenceBadge.hidden = false;
    confidenceValue.textContent = `${Math.round(data.confidence * 100)}%`;

    renderedEq.innerHTML = `\\[${data.latex}\\]`;
    if (latexOutput) latexOutput.textContent = data.latex;

    solutionOutput.innerHTML = "";
    currentSolutionText = "";
    if (data.solution) {
      const sol = data.solution;
      const metaDiv = document.createElement("div");
      metaDiv.className = "sol-meta";
      const lines = [];
      lines.push(`Kind: ${sol.kind} | Input: ${sol.sympy_input}`);
      if (sol.kind === "equation") {
        lines.push(
          sol.solutions && sol.solutions.length
            ? `Solutions: ${sol.solutions.join(", ")}`
            : "No closed-form solutions found."
        );
      } else {
        lines.push(`Simplified: ${sol.simplified}`);
        if (sol.numeric_value) lines.push(`Numeric value: ${sol.numeric_value}`);
      }
      metaDiv.textContent = lines.join("\n");
      solutionOutput.appendChild(metaDiv);

      const copyLines = [...lines];

      if (sol.steps && sol.steps.length > 0) {
        const stepsWrapper = document.createElement("div");
        stepsWrapper.className = "solution-steps-wrapper";
        const stepsHeading = document.createElement("h4");
        stepsHeading.textContent = "Step-by-Step Derivation";
        stepsWrapper.appendChild(stepsHeading);

        const ol = document.createElement("ol");
        ol.className = "steps-list";
        copyLines.push("\nStep-by-Step Derivation:");
        sol.steps.forEach((stepTex, idx) => {
          const li = document.createElement("li");
          li.className = "step-item";
          li.innerHTML = `\\[${stepTex}\\]`;
          ol.appendChild(li);

          let cleanStep = stepTex
            .replace(/\\text\{([^}]+)\}/g, "$1")
            .replace(/\\quad/g, " ")
            .replace(/\\!/g, "")
            .replace(/\s+/g, " ")
            .trim();
          copyLines.push(`${idx + 1}. ${cleanStep}`);
        });
        stepsWrapper.appendChild(ol);
        solutionOutput.appendChild(stepsWrapper);
      }

      currentSolutionText = copyLines.join("\n");
    } else if (data.solve_warning) {
      solutionOutput.innerHTML = `<span class="warning">${data.solve_warning}</span>`;
      currentSolutionText = data.solve_warning;
    }

    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([renderedEq, solutionOutput]).catch(() => {});
    }

    samplesList.innerHTML = "";
    (data.samples || []).forEach((s) => {
      const li = document.createElement("li");
      li.textContent = s;
      samplesList.appendChild(li);
    });
  }

  // ------------------------------------------------------------------
  // Clipboard Copy Handling
  // ------------------------------------------------------------------
  function handleCopy(btn, textToCopy, defaultLabel) {
    if (!textToCopy) return;
    const labelEl = btn.querySelector(".copy-label");

    const onSuccess = () => {
      btn.classList.add("copied");
      if (labelEl) labelEl.textContent = "Copied!";
      setTimeout(() => {
        btn.classList.remove("copied");
        if (labelEl) labelEl.textContent = defaultLabel;
      }, 2000);
    };

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(textToCopy).then(onSuccess).catch(() => {
        fallbackCopy(textToCopy, onSuccess);
      });
    } else {
      fallbackCopy(textToCopy, onSuccess);
    }
  }

  function fallbackCopy(text, cb) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
      cb();
    } catch (_e) {}
    document.body.removeChild(ta);
  }

  if (copyLatexBtn) {
    copyLatexBtn.addEventListener("click", () => {
      handleCopy(copyLatexBtn, latexOutput ? latexOutput.textContent : "", "Copy");
    });
  }

  if (copySolutionBtn) {
    copySolutionBtn.addEventListener("click", () => {
      handleCopy(copySolutionBtn, currentSolutionText, "Copy solution");
    });
  }
})();
