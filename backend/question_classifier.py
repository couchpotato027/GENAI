"""
Question Classifier
====================
Classifies exam questions into three psychometric categories using
student performance statistics. This shared module is used by all
four agents to ensure consistent interpretation.

Categories & Tiered Quality Labels
-----------------------------------
  "easy"       — Case C: "Suitable for Recall Assessment"
                 High avg, high pass rate. Low discrimination is EXPECTED
                 for recall-level questions and does not indicate flawed design.

  "hard_valid" — Case B: "Good Discriminator"
                 Low avg, low pass rate, high variance. The question
                 effectively separates students by ability — positive
                 discrimination.

  "poor"       — Case A: "Poor Discriminator (Ambiguous Design)"
                 Mid-range avg with low variance, or low avg with low
                 variance. Scores cluster near the mean, suggesting
                 ambiguous wording or misalignment.

Academic Rationale — Why Easy Questions Can Have Low Discrimination
-------------------------------------------------------------------
In classical test theory, discrimination (Point-Biserial Correlation)
measures how well an item separates high-ability from low-ability
students. Easy questions inherently produce low discrimination because
MOST students answer correctly, leaving insufficient variance for
differentiation. This is not a design flaw — it reflects the intended
assessment purpose:

  - Recall / knowledge check → Easy is appropriate
  - Formative assessment     → Easy items verify baseline understanding
  - Ability discrimination   → Harder items with variance are needed

Low discrimination ≠ Bad question. It means the question serves a
different assessment purpose (recall vs. discrimination).
"""

# ── Configurable thresholds (adjust per institutional norms) ──
EASY_AVG_THRESHOLD   = 70     # above this → likely easy
EASY_PASS_THRESHOLD  = 80     # above this → most students pass
POOR_AVG_LOW         = 40     # scores clustered in this band…
POOR_AVG_HIGH        = 60     # …indicate ambiguity
HARD_AVG_THRESHOLD   = 45     # below this → likely hard
HARD_PASS_THRESHOLD  = 50     # below this → low pass rate
VARIANCE_BOUNDARY    = 100    # above → high spread (discrimination)

# ── Tiered quality labels ──
QUALITY_LABELS = {
    "easy":       "Suitable for Recall Assessment",
    "hard_valid": "Good Discriminator",
    "poor":       "Poor Discriminator (Ambiguous Design)",
}


def classify_question(avg_score: float, pass_rate: float, variance: float) -> str:
    """
    Classify a question based on student performance statistics.

    Returns one of: "poor", "hard_valid", "easy".
    """
    if avg_score >= EASY_AVG_THRESHOLD and pass_rate >= EASY_PASS_THRESHOLD:
        return "easy"
    if avg_score < HARD_AVG_THRESHOLD and pass_rate < HARD_PASS_THRESHOLD and variance > VARIANCE_BOUNDARY:
        return "hard_valid"
    return "poor"


def get_quality_label(case: str) -> str:
    """Return the human-readable tiered quality label for a case."""
    return QUALITY_LABELS.get(case, "Unknown")


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        {"avg": 85.5, "pass": 100.0, "var": 37.2,  "label": "Easy recall (85,90,78…)"},
        {"avg": 52.5, "pass": 50.0,  "var": 2.9,   "label": "Clustered mid-scores (48,52,50,49,51)"},
        {"avg": 35.0, "pass": 33.3,  "var": 1450.0,"label": "Low avg + high variance (15,85,20,90)"},
        {"avg": 30.0, "pass": 25.0,  "var": 50.0,  "label": "Low avg + low variance (floor effect)"},
    ]

    print("Question Classifier — Demo\n" + "=" * 50)
    for c in cases:
        case = classify_question(c["avg"], c["pass"], c["var"])
        print(f"\n{c['label']}")
        print(f"  avg={c['avg']}, pass={c['pass']}%, var={c['var']}")
        print(f"  → Case: {case}")
        print(f"  → Quality: {get_quality_label(case)}")
    print("=" * 50)
