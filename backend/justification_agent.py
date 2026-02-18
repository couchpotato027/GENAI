"""
Justification Agent
===================
Explains the rationale behind assessment quality classification and
suggested improvements, using calibrated academic language.

Key principles:
  - Clearly distinguishes assessment intent, difficulty, and discrimination
  - Never uses absolute negative language ("fails", "incorrect", "bad")
  - Uses calibrated phrasing ("better suited for", "may not be optimal for")
  - Explains why low discrimination in easy questions is expected, not a flaw
"""

from question_classifier import classify_question


class JustificationAgent:
    def __init__(self):
        self._justifications = {
            "easy": {
                "discrimination": (
                    "Low discrimination is expected for recall-level questions. "
                    "When most students answer correctly, there is insufficient "
                    "variance to separate ability levels — this is a natural "
                    "property of easy items, not a design deficiency. The question "
                    "is better suited for formative assessment and knowledge "
                    "verification than for ability discrimination."
                ),
                "difficulty": (
                    "The difficulty level is appropriate for the assessment purpose. "
                    "If the intent is to verify baseline knowledge, the current "
                    "difficulty is well-calibrated. If higher-order thinking is the "
                    "goal, the optional rewrite elevates cognitive demand while "
                    "preserving the core content."
                ),
                "learning_outcomes": (
                    "The question aligns with lower-order learning outcomes "
                    "(e.g., 'Remember', 'Identify'). If the course objective "
                    "requires higher-order outcomes (e.g., 'Analyze', 'Evaluate'), "
                    "the suggested upgrade provides a path to better alignment."
                ),
            },
            "hard_valid": {
                "discrimination": (
                    "The question demonstrates positive discrimination: high "
                    "variance in student scores indicates effective separation "
                    "between students who have mastered the material and those "
                    "who have not. This is a desirable psychometric property "
                    "for summative assessment."
                ),
                "difficulty": (
                    "The difficulty level is appropriate for the intended "
                    "assessment. The low average score reflects genuine content "
                    "complexity rather than ambiguity. The suggested structural "
                    "improvements (sub-parts, explicit assumptions) preserve "
                    "rigor while giving students a clearer path to demonstrate "
                    "partial mastery."
                ),
                "learning_outcomes": (
                    "By adding sub-parts and explicit mark allocation, the "
                    "revised question allows students to demonstrate progressive "
                    "understanding — measuring what students know rather than "
                    "only penalizing what they don't."
                ),
            },
            "poor": {
                "discrimination": (
                    "The question may not be optimal for ability discrimination "
                    "— scores cluster near the mean with low variance, meaning "
                    "students of different ability levels respond similarly. "
                    "This could result from ambiguous wording, misaligned "
                    "distractors, or content that doesn't clearly target the "
                    "intended learning outcome."
                ),
                "difficulty": (
                    "The difficulty level may be better calibrated by ensuring "
                    "that the challenge comes from conceptual depth rather than "
                    "from unclear phrasing. The revision aims to make the "
                    "difficulty source transparent and intentional."
                ),
                "learning_outcomes": (
                    "By improving the clarity of the question stem and aligning "
                    "it more closely with specific learning outcomes, the revised "
                    "question better measures the intended competency without "
                    "introducing construct-irrelevant difficulty."
                ),
            },
        }

    def justify(self, original: str, improved: str, issues: str,
                principles: str, stats: dict = None) -> str:
        """
        Generate a justification distinguishing assessment intent,
        difficulty, and discrimination.
        """
        case = "poor"
        if stats:
            case = classify_question(
                stats.get('avg_score', 50),
                stats.get('pass_rate', 50),
                stats.get('variance', 0)
            )

        justification = self._justifications.get(case)
        if not justification:
            return (
                "Justification:\n"
                "- The question appears well-constructed for its intended "
                "assessment purpose. Verify alignment with learning objectives."
            )

        output_lines = [
            "Justification:",
            "- Discrimination analysis:",
            f"  {justification['discrimination']}",
            "- Difficulty calibration:",
            f"  {justification['difficulty']}",
            "- Learning outcome alignment:",
            f"  {justification['learning_outcomes']}",
        ]

        return "\n".join(output_lines)


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    agent = JustificationAgent()

    test_cases = [
        {
            "label": "Easy recall",
            "original": "What is photosynthesis?",
            "improved": "Explain the significance of photosynthesis…",
            "issues": "Expected for recall-level questions...",
            "principles": "Formative assessment",
            "stats": {"avg_score": 92, "pass_rate": 95, "variance": 30}
        },
        {
            "label": "Good discriminator",
            "original": "Derive the eigenvalue decomposition…",
            "improved": "(a) Define… (b) Compute… (c) Derive…",
            "issues": "High variance... positive discrimination...",
            "principles": "Scaffolding, Sub-parts",
            "stats": {"avg_score": 35, "pass_rate": 33, "variance": 1450}
        },
        {
            "label": "Poor discriminator",
            "original": "Select the synonym for 'Happy'.",
            "improved": "Clarify specific constraints…",
            "issues": "Low variance... clustering...",
            "principles": "Clarity, Alignment",
            "stats": {"avg_score": 52, "pass_rate": 50, "variance": 3}
        },
    ]

    print("Justification Agent — Demo\n" + "=" * 50)
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Case {i}] {case['label']}")
        print("-" * 30)
        print(agent.justify(
            case["original"], case["improved"],
            case["issues"], case["principles"], case["stats"]
        ))
        print("=" * 50)
