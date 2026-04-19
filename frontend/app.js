/**
 * Intelligent Exam Question Analysis & AI Assessment Assistant
 * Frontend Logic — Milestone 1 (Classical ML) + Milestone 2 (Agentic AI)
 */

// Auto-detect: in local dev, talk directly to backend; in production, use Vercel proxy
const API_BASE_URL = window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "/api";

// State
const state = {
    currentMode: "analytics",
    analysisComplete: false,
    analysisData: null,
    agentRunning: false,
    agentData: null,
};

// DOM References
const dom = {
    // Nav
    btnAnalytics: document.getElementById("btnAnalytics"),
    btnAgentic: document.getElementById("btnAgentic"),
    btnAboutUs: document.getElementById("btnAboutUs"),

    // Panels
    analyticsPanel: document.getElementById("analyticsPanel"),
    agenticPanel: document.getElementById("agenticPanel"),
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

    // Agentic Panel
    agentNoData: document.getElementById("agentNoData"),
    agentContext: document.getElementById("agentContext"),
    agentQuestion: document.getElementById("agentQuestion"),
    agentDifficulty: document.getElementById("agentDifficulty"),
    agentConfidence: document.getElementById("agentConfidence"),
    agentAvgScore: document.getElementById("agentAvgScore"),
    agentPassRate: document.getElementById("agentPassRate"),
    scoreDominanceNotice: document.getElementById("scoreDominanceNotice"),
    btnRunAgent: document.getElementById("btnRunAgent"),

    // Agent Progress
    agentProgressSection: document.getElementById("agentProgressSection"),
    agentProgressFill: document.getElementById("agentProgressFill"),
    agentProgressLabel: document.getElementById("agentProgressLabel"),

    // Agent Results
    agentResultsSection: document.getElementById("agentResultsSection"),
    agentSummaryText: document.getElementById("agentSummaryText"),
    agentDifficultyText: document.getElementById("agentDifficultyText"),
    agentGapsList: document.getElementById("agentGapsList"),
    agentIssuesList: document.getElementById("agentIssuesList"),
    agentRecsList: document.getElementById("agentRecsList"),
    agentRefsList: document.getElementById("agentRefsList"),
    agentEthicalNotice: document.getElementById("agentEthicalNotice"),
    agentEthicalText: document.getElementById("agentEthicalText"),
};


// ═══════════════════════════════════════════════
// PAGE SWITCHING (Analytics / Agentic / About Us)
// ═══════════════════════════════════════════════
function switchMode(mode) {
    state.currentMode = mode;

    dom.btnAnalytics.classList.toggle("active", mode === "analytics");
    dom.btnAgentic.classList.toggle("active", mode === "agentic");
    dom.btnAboutUs.classList.toggle("active", mode === "about");

    dom.analyticsPanel.style.display = mode === "analytics" ? "block" : "none";
    dom.agenticPanel.style.display = mode === "agentic" ? "block" : "none";
    dom.aboutPanel.style.display = mode === "about" ? "block" : "none";

    // When switching to agentic, update context from ML analysis
    if (mode === "agentic") {
        updateAgenticContext();
    }
}

dom.btnAnalytics.addEventListener("click", () => switchMode("analytics"));
dom.btnAgentic.addEventListener("click", () => switchMode("agentic"));
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
        scores: scoresStr,
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


// ═══════════════════════════════════════════════
// AGENTIC AI ASSISTANT (MILESTONE 2)
// ═══════════════════════════════════════════════

function updateAgenticContext() {
    if (!state.analysisComplete || !state.analysisData) {
        dom.agentNoData.style.display = "block";
        dom.agentContext.style.display = "none";
        return;
    }

    const d = state.analysisData;
    dom.agentNoData.style.display = "none";
    dom.agentContext.style.display = "block";

    dom.agentQuestion.textContent = d.question || "(no question text)";
    dom.agentDifficulty.textContent = d.difficulty;
    dom.agentConfidence.textContent = (d.confidence * 100).toFixed(0) + "%";
    dom.agentAvgScore.textContent = d.avgScore;
    dom.agentPassRate.textContent = d.rateValue + "%";

    // Score dominance is always true for this model
    dom.scoreDominanceNotice.style.display = "block";
}

// Agent Card Expand/Collapse
document.querySelectorAll(".agent-header").forEach(header => {
    header.addEventListener("click", () => {
        const targetId = header.getAttribute("data-target");
        if (!targetId) return;
        const body = document.getElementById(targetId);
        if (!body) return;
        const isCollapsed = body.classList.contains("collapsed");
        body.classList.toggle("collapsed", !isCollapsed);
        const icon = header.querySelector(".agent-expand");
        if (icon) icon.classList.toggle("open", isCollapsed);
    });
});


// ═══════════════════════════════════════════════
// AGENT PIPELINE EXECUTION
// ═══════════════════════════════════════════════

const AGENT_STEPS = [
    { label: "Validating inputs...", progress: 10 },
    { label: "Running ML prediction...", progress: 25 },
    { label: "Interpreting results...", progress: 40 },
    { label: "Retrieving pedagogy documents (RAG)...", progress: 55 },
    { label: "Generating AI recommendations (LLM)...", progress: 75 },
    { label: "Formatting final report...", progress: 90 },
    { label: "Complete!", progress: 100 },
];

async function simulateProgress() {
    dom.agentProgressSection.style.display = "block";
    dom.agentResultsSection.style.display = "none";

    for (const step of AGENT_STEPS) {
        dom.agentProgressFill.style.width = step.progress + "%";
        dom.agentProgressLabel.textContent = step.label;
        await new Promise(r => setTimeout(r, 400));
    }
}

dom.btnRunAgent.addEventListener("click", async () => {
    if (state.agentRunning) return;
    if (!state.analysisComplete || !state.analysisData) return;

    state.agentRunning = true;
    dom.btnRunAgent.disabled = true;
    dom.btnRunAgent.textContent = "Analyzing...";

    // Start progress animation
    const progressPromise = simulateProgress();

    try {
        const d = state.analysisData;

        const response = await fetch(`${API_BASE_URL}/agent/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: d.question,
                student_scores: d.scores,
            }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(err.detail || response.statusText);
        }

        const result = await response.json();
        state.agentData = result;

        // Wait for progress animation to finish
        await progressPromise;

        // Render results
        renderAgentResults(result);

    } catch (err) {
        await progressPromise;
        dom.agentProgressLabel.textContent = `Error: ${err.message}`;
        dom.agentProgressFill.style.width = "100%";
        dom.agentProgressFill.style.background = "var(--accent-red)";
    } finally {
        state.agentRunning = false;
        dom.btnRunAgent.disabled = false;
        dom.btnRunAgent.innerHTML = '<span class="btn-icon">&#9733;</span> Run AI Assessment Analysis';
    }
});


function renderAgentResults(data) {
    dom.agentResultsSection.style.display = "block";
    dom.agentProgressSection.style.display = "none";

    // Reset progress bar color
    dom.agentProgressFill.style.background = "";

    // Summary
    dom.agentSummaryText.textContent = data.summary || "No summary available.";

    // Difficulty Analysis
    dom.agentDifficultyText.textContent = data.difficulty_analysis || "No analysis available.";

    // Learning Gaps
    renderList(dom.agentGapsList, data.learning_gaps);

    // Question Issues
    renderList(dom.agentIssuesList, data.question_issues);

    // Recommendations
    renderList(dom.agentRecsList, data.recommendations);

    // Pedagogical References
    renderList(dom.agentRefsList, data.pedagogical_references);

    // Score Dominance highlight
    if (data.score_dominance) {
        dom.scoreDominanceNotice.style.display = "block";
    }

    // Ethical Notice
    if (data.ethical_notice) {
        dom.agentEthicalNotice.style.display = "flex";
        dom.agentEthicalText.textContent = data.ethical_notice;
    }
}

function renderList(ul, items) {
    ul.innerHTML = "";
    if (!items || !Array.isArray(items) || items.length === 0) {
        const li = document.createElement("li");
        li.textContent = "No items identified.";
        ul.appendChild(li);
        return;
    }
    items.forEach(item => {
        const li = document.createElement("li");
        li.textContent = item;
        ul.appendChild(li);
    });
}
