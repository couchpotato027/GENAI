"""
Assessment Analysis Agent
=========================
Analyzes exam questions using the Case A/B/C classifier and produces
tiered quality labels instead of binary Good/Poor.

Tiered Labels:
  "Suitable for Recall Assessment"       — Easy, high pass rate
  "Good Discriminator"                   — Hard, high variance
  "Poor Discriminator (Ambiguous Design)" — Clustered scores, low variance
"""

from question_classifier import classify_question, get_quality_label


class AssessmentAnalysisAgent:
    def __init__(self):
        pass

    def analyze(self, question_text: str, predicted_difficulty: str,
                discrimination_index: float, stats: dict) -> str:
        """
        Analyze the quality of an exam question.

        Parameters
        ----------
        question_text        : The content of the question.
        predicted_difficulty : 'Easy', 'Medium', or 'Hard'.
        discrimination_index : Float between -1.0 and 1.0.
        stats                : Dict with keys 'avg_score', 'pass_rate', 'variance'.

        Returns
        -------
        Formatted string with Assessment Quality (tiered) and Observations.
        """
        observations = []

        avg  = stats.get('avg_score', 0)
        pr   = stats.get('pass_rate', 0)
        var  = stats.get('variance', 0)

        # ── Variance-aware classification ──
        case = classify_question(avg, pr, var)
        quality = get_quality_label(case)

        if case == "easy":
            # Case C: Easy / Recall-suitable
            observations.append(
                f"High pass rate ({pr:.1f}%) with low variance ({var:.1f}) "
                "is expected for recall-level questions. Low discrimination "
                "does not indicate flawed design — it reflects the assessment "
                "purpose (knowledge check / formative assessment)."
            )
            if pr > 95 and predicted_difficulty != "Easy":
                observations.append(
                    f"Note: {pr:.1f}% of students passed, which may indicate "
                    "a ceiling effect if the intended goal is ability discrimination."
                )

        elif case == "hard_valid":
            # Case B: Hard but well-designed — positive discrimination
            observations.append(
                f"High variance ({var:.1f}) on a difficult question indicates "
                "positive discrimination — the question effectively separates "
                "students who have mastered the material from those who have not. "
                "This is a desirable psychometric property for summative assessment."
            )

        else:
            # Case A: Poor discriminator — ambiguous design
            if discrimination_index < 0:
                observations.append(
                    "Negative discrimination detected: high-performing students "
                    "are more likely to answer incorrectly than low-performing "
                    "students. This may indicate ambiguous wording, an incorrect "
                    "answer key, or a 'trick' element that penalizes deeper reasoning."
                )
            else:
                observations.append(
                    f"Low variance ({var:.1f}) with scores clustering near the "
                    f"mean ({avg:.1f}) suggests the question may not be optimally "
                    "worded. Students of different ability levels are responding "
                    "similarly, which limits the question's discriminatory value."
                )

            # Difficulty-mismatch checks
            if predicted_difficulty == "Easy" and pr < 50:
                observations.append(
                    f"Content mismatch: Predicted 'Easy' but only "
                    f"{pr:.1f}% passed. The question may be better suited for "
                    "a higher difficulty tier, or the wording may be misleading."
                )
            if predicted_difficulty == "Hard" and pr > 80:
                observations.append(
                    f"Content mismatch: Predicted 'Hard' but "
                    f"{pr:.1f}% passed. The question may be better suited for "
                    "a lower difficulty tier."
                )

        # ── Format output ──
        output_lines = [
            f"Assessment Quality: {quality}",
            "Observations:"
        ]
        for obs in observations:
            output_lines.append(f"- {obs}")

        return "\n".join(output_lines)


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    agent = AssessmentAnalysisAgent()

    test_cases = [
        {
            "label": "Easy recall question",
            "text": "What is the capital of France?",
            "diff": "Easy", "disc": 0.45,
            "stats": {"avg_score": 92, "pass_rate": 95, "variance": 30}
        },
        {
            "label": "Hard, well-discriminating question",
            "text": "Derive the eigenvalue decomposition…",
            "diff": "Hard", "disc": -0.15,
            "stats": {"avg_score": 35, "pass_rate": 33, "variance": 1450}
        },
        {
            "label": "Ambiguous, clustered mid-scores",
            "text": "Select the synonym for 'Happy'.",
            "diff": "Easy", "disc": 0.05,
            "stats": {"avg_score": 52, "pass_rate": 50, "variance": 3}
        },
    ]

    print("Assessment Analysis Agent — Demo\n" + "=" * 50)
    for i, case in enumerate(test_cases, 1):
        print(f"\nCase {i}: {case['label']}")
        print(f"  Q: {case['text']}")
        print("-" * 30)
        print(agent.analyze(case["text"], case["diff"], case["disc"], case["stats"]))
        print("=" * 50)
