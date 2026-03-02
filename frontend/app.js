/**
 * ═══════════════════════════════════════════════════════
 * Intelligent Exam Question Analysis
 * Frontend Logic — Milestone 1 (Classical ML)
 * ═══════════════════════════════════════════════════════
 */

const API_BASE_URL = "https://genai-ztl4.onrender.com";

// ─── State ───
const state = {
    analysisComplete: false,
    analysisData: null,
};

// ─── DOM References ───
const dom = {
    // Panels
    analyticsPanel: document.getElementById("analyticsPanel"),

    // Input
    questionText: document.getElementById("questionText"),
    csvUpload: document.getElementById("csvUpload"),
    fileUploadArea: document.getElementById("fileUploadArea"),
    uploadPlaceholder: document.getElementById("uploadPlaceholder"),
    uploadSuccess: document.getElementById("uploadSuccess"),
    fileName: document.getElementById("fileName"),
    removeFile: document.getElementById("removeFile"),
    browseLink: document.getElementById("browseLink"),
    manualScores: document.getElementById("manualScores"),
    btnRunAnalysis: document.getElementById("btnRunAnalysis"),

    // Results
    resultsSection: document.getElementById("resultsSection"),
    resDifficulty: document.getElementById("resDifficulty"),
    resConfidence: document.getElementById("resConfidence"),
    confidenceFill: document.getElementById("confidenceFill"),
    resAvgScore: document.getElementById("resAvgScore"),
    resPctCorrect: document.getElementById("resPctCorrect"),
    resVariance: document.getElementById("resVariance"),

    // Evaluation
    evalSection: document.getElementById("evalSection"),
    evalToggle: document.getElementById("evalToggle"),
    evalContent: document.getElementById("evalContent"),
    expandIcon: document.getElementById("expandIcon"),
    evalAccuracy: document.getElementById("evalAccuracy"),
    evalPrecision: document.getElementById("evalPrecision"),
    evalRecall: document.getElementById("evalRecall"),
};


// ═══════════════════════════════════════════════
// FILE UPLOAD
// ═══════════════════════════════════════════════
dom.browseLink.addEventListener("click", (e) => {
    e.preventDefault();
    dom.csvUpload.click();
});

dom.fileUploadArea.addEventListener("click", () => {
    if (dom.uploadSuccess.style.display === "none") {
        dom.csvUpload.click();
    }
});

dom.csvUpload.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        showFile(e.target.files[0].name);
    }
});

dom.fileUploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dom.fileUploadArea.classList.add("dragover");
});

dom.fileUploadArea.addEventListener("dragleave", () => {
    dom.fileUploadArea.classList.remove("dragover");
});

dom.fileUploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dom.fileUploadArea.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        showFile(e.dataTransfer.files[0].name);
    }
});

function showFile(name) {
    dom.uploadPlaceholder.style.display = "none";
    dom.uploadSuccess.style.display = "flex";
    dom.fileName.textContent = name;
}

dom.removeFile.addEventListener("click", (e) => {
    e.stopPropagation();
    dom.csvUpload.value = "";
    dom.uploadPlaceholder.style.display = "block";
    dom.uploadSuccess.style.display = "none";
    dom.fileName.textContent = "";
});


// ═══════════════════════════════════════════════
// ML ANALYSIS — Strict Validation & Computation
// ═══════════════════════════════════════════════

/**
 * Returns true only when the user has supplied real student score data.
 * Accepts either a non-empty manual scores string or a CSV file upload.
 */
function hasStudentData() {
    const manualFilled = dom.manualScores.value.trim().length > 0;
    const csvUploaded = dom.csvUpload.files && dom.csvUpload.files.length > 0;
    return manualFilled || csvUploaded;
}

/**
 * Show or hide the inline validation warning.
 */
function setWarning(message) {
    let warningEl = document.getElementById("dataWarning");
    if (!warningEl) return;
    if (message) {
        warningEl.textContent = message;
        warningEl.style.display = "block";
    } else {
        warningEl.textContent = "";
        warningEl.style.display = "none";
    }
}

/**
 * Update the Run ML Analysis button's disabled state based on
 * whether student data is currently present.
 */
function refreshButtonState() {
    const dataPresent = hasStudentData();
    dom.btnRunAnalysis.disabled = !dataPresent;
    if (dataPresent) {
        setWarning(null);
    }
}

/**
 * Parse and validate the raw scores string.
 * Throws a descriptive Error if the data is missing or unparseable.
 * IMPORTANT: No fallback values are ever generated here.
 */
function parseScores(scoresStr) {
    const scores = scoresStr
        .split(",")
        .map(s => parseFloat(s.trim()))
        .filter(n => !isNaN(n) && n >= 0 && n <= 100);

    if (scores.length === 0) {
        throw new Error("No valid numeric scores found. Scores must be comma-separated numbers between 0 and 100.");
    }
    if (scores.length < 3) {
        throw new Error(`Only ${scores.length} valid score(s) provided. At least 3 student responses are required for meaningful analysis.`);
    }
    return scores;
}

/**
 * Detect whether scores are binary (0/1 only) or continuous (0–100 marks).
 * Returns true if every score is exactly 0 or exactly 1.
 */
function isBinary(scores) {
    return scores.every(s => s === 0 || s === 1);
}

/**
 * Configurable pass threshold for continuous scores.
 * Faculty can adjust this value as needed.
 */
const PASS_THRESHOLD = 50;

/**
 * Run Analysis via Backend API
 */
async function runAnalysis(questionText, scoresStr) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: questionText, student_scores: scoresStr })
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        const res = await response.json();

        // Map backend response to frontend data model
        return {
            question: questionText,
            difficulty: res.predicted_difficulty,
            confidence: res.confidence,
            avgScore: res.avg_score.toFixed(1),
            rateValue: res.pass_rate.toFixed(1),
            metricLabel: "Pass Rate (≥50%)",
            variance: res.variance.toFixed(1),
            discIndex: res.disc_index
        };
    } catch (err) {
        throw err;
    }
}


// ═══════════════════════════════════════════════
// OFFLINE MODEL EVALUATION (Static Constants)
// These are computed once during model training/validation
// and NEVER recomputed from live user input.
// ═══════════════════════════════════════════════
const OFFLINE_MODEL_EVAL = Object.freeze({
    accuracy: "0.9500",
    precision: "0.9480",
    recall: "0.9500",
    confusionMatrix: Object.freeze([
        [28, 0, 0],
        [1, 32, 1],
        [0, 2, 16],
    ]),
});

/**
 * Populate the evaluation UI once at page load from
 * the frozen offline constants.
 */
function populateOfflineEval() {
    dom.evalAccuracy.textContent = OFFLINE_MODEL_EVAL.accuracy;
    dom.evalPrecision.textContent = OFFLINE_MODEL_EVAL.precision;
    dom.evalRecall.textContent = OFFLINE_MODEL_EVAL.recall;

    const cm = OFFLINE_MODEL_EVAL.confusionMatrix;
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            const cell = document.getElementById(`cm${i}${j}`);
            cell.textContent = cm[i][j];
            cell.classList.toggle("diagonal", i === j);
        }
    }
}

// Initialise offline eval values immediately
populateOfflineEval();

// ─── Live button-state watcher ───
// Disable the button whenever student data is absent.
dom.manualScores.addEventListener("input", refreshButtonState);
dom.csvUpload.addEventListener("change", refreshButtonState);
dom.removeFile.addEventListener("click", () => setTimeout(refreshButtonState, 0));

// Initialise: button starts disabled until data is provided
refreshButtonState();

// ─── Run ML Analysis click handler ───
dom.btnRunAnalysis.addEventListener("click", async () => {
    const question = dom.questionText.value.trim();

    // Guard 1: question text is mandatory
    if (!question) {
        setWarning("Exam question text is required.");
        dom.questionText.focus();
        return;
    }

    // Guard 2: student data is mandatory — no silent fallback
    if (!hasStudentData()) {
        setWarning("Student response data is required to run ML analysis.");
        return;
    }

    // Determine which source to use (manual takes priority over CSV for now)
    const scoresStr = dom.manualScores.value.trim();

    // UI Loading State
    const originalText = dom.btnRunAnalysis.textContent;
    dom.btnRunAnalysis.textContent = "Analyzing...";
    dom.btnRunAnalysis.disabled = true;

    try {
        const data = await runAnalysis(question, scoresStr);

        // All guards passed — clear any previous warning
        setWarning(null);

        state.analysisData = data;
        state.analysisComplete = true;

        // Populate results
        dom.resDifficulty.textContent = data.difficulty;
        dom.resDifficulty.className = "card-value difficulty-badge " + data.difficulty.toLowerCase();
        dom.resConfidence.textContent = (data.confidence * 100).toFixed(0) + "%";
        dom.confidenceFill.style.width = (data.confidence * 100) + "%";
        dom.resAvgScore.textContent = data.avgScore;

        // Dynamic label
        document.getElementById("resPctLabel").textContent = data.metricLabel;
        dom.resPctCorrect.textContent = data.rateValue + "%";
        dom.resVariance.textContent = data.variance;

        // Show sections (eval metrics are already populated from offline constants)
        dom.resultsSection.style.display = "block";
        dom.evalSection.style.display = "block";

    } catch (err) {
        setWarning(`Runtime Error: ${err.message}`);
    } finally {
        dom.btnRunAnalysis.textContent = originalText;
        dom.btnRunAnalysis.disabled = false;
    }
});


// ═══════════════════════════════════════════════
// EVALUATION EXPANDABLE
// ═══════════════════════════════════════════════
dom.evalToggle.addEventListener("click", () => {
    const isOpen = dom.evalContent.style.display !== "none";
    dom.evalContent.style.display = isOpen ? "none" : "block";
    dom.expandIcon.classList.toggle("open", !isOpen);
});
