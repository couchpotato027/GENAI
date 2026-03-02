/**
 * Intelligent Exam Question Analysis
 * Frontend Logic — Milestone 1 (Classical ML)
 */

const API_BASE_URL = "https://genai-ztl4.onrender.com";

// State
const state = {
    currentMode: "analytics",
    analysisComplete: false,
    analysisData: null,
};

// DOM References
const dom = {
    // Nav
    btnAnalytics: document.getElementById("btnAnalytics"),
    btnAboutUs: document.getElementById("btnAboutUs"),

    // Panels
    analyticsPanel: document.getElementById("analyticsPanel"),
    aboutPanel: document.getElementById("aboutPanel"),

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
// PAGE SWITCHING (Analytics / About Us)
// ═══════════════════════════════════════════════
function switchMode(mode) {
    state.currentMode = mode;

    dom.btnAnalytics.classList.toggle("active", mode === "analytics");
    dom.btnAboutUs.classList.toggle("active", mode === "about");

    dom.analyticsPanel.style.display = mode === "analytics" ? "block" : "none";
    dom.aboutPanel.style.display = mode === "about" ? "block" : "none";
}

dom.btnAnalytics.addEventListener("click", () => switchMode("analytics"));
dom.btnAboutUs.addEventListener("click", () => switchMode("about"));


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
// ML ANALYSIS — Validation
// ═══════════════════════════════════════════════

function hasStudentData() {
    const manualFilled = dom.manualScores.value.trim().length > 0;
    const csvUploaded = dom.csvUpload.files && dom.csvUpload.files.length > 0;
    return manualFilled || csvUploaded;
}

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

function refreshButtonState() {
    const dataPresent = hasStudentData();
    dom.btnRunAnalysis.disabled = !dataPresent;
    if (dataPresent) {
        setWarning(null);
    }
}

function parseScores(scoresStr) {
    const scores = scoresStr
        .split(",")
        .map(s => parseFloat(s.trim()))
        .filter(n => !isNaN(n) && n >= 0 && n <= 100);

    if (scores.length === 0) {
        throw new Error("No valid numeric scores found. Scores must be comma-separated numbers between 0 and 100.");
    }
    if (scores.length < 3) {
        throw new Error(`Only ${scores.length} valid score(s) provided. At least 3 student responses are required.`);
    }
    return scores;
}

function isBinary(scores) {
    return scores.every(s => s === 0 || s === 1);
}

const PASS_THRESHOLD = 50;

async function runAnalysis(questionText, scoresStr) {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: questionText, student_scores: scoresStr })
    });

    if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
    }

    const res = await response.json();

    return {
        question: questionText,
        difficulty: res.predicted_difficulty,
        confidence: res.confidence,
        avgScore: res.avg_score.toFixed(1),
        rateValue: res.pass_rate.toFixed(1),
        metricLabel: "Pass Rate",
        variance: res.variance.toFixed(1),
        discIndex: res.disc_index
    };
}


// ═══════════════════════════════════════════════
// OFFLINE MODEL EVALUATION (Static)
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

populateOfflineEval();

// Live button-state watcher
dom.manualScores.addEventListener("input", refreshButtonState);
dom.csvUpload.addEventListener("change", refreshButtonState);
dom.removeFile.addEventListener("click", () => setTimeout(refreshButtonState, 0));
refreshButtonState();

// Run ML Analysis click handler
dom.btnRunAnalysis.addEventListener("click", async () => {
    const question = dom.questionText.value.trim();

    if (!question) {
        setWarning("Exam question text is required.");
        dom.questionText.focus();
        return;
    }

    if (!hasStudentData()) {
        setWarning("Student response data is required to run ML analysis.");
        return;
    }

    const scoresStr = dom.manualScores.value.trim();

    const originalText = dom.btnRunAnalysis.textContent;
    dom.btnRunAnalysis.textContent = "Analyzing...";
    dom.btnRunAnalysis.disabled = true;

    try {
        const data = await runAnalysis(question, scoresStr);
        setWarning(null);

        state.analysisData = data;
        state.analysisComplete = true;

        dom.resDifficulty.textContent = data.difficulty;
        dom.resDifficulty.className = "card-value difficulty-badge " + data.difficulty.toLowerCase();
        dom.resConfidence.textContent = (data.confidence * 100).toFixed(0) + "%";
        dom.confidenceFill.style.width = (data.confidence * 100) + "%";
        dom.resAvgScore.textContent = data.avgScore;

        document.getElementById("resPctLabel").textContent = data.metricLabel;
        dom.resPctCorrect.textContent = data.rateValue + "%";
        dom.resVariance.textContent = data.variance;

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
