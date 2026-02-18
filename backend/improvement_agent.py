"""
Assessment Improvement Agent
============================
Proposes structured improvements based on the question's psychometric
classification. Uses calibrated language:

  Easy (recall)    → Optional upgrades, not fixes
  Hard (valid)     → Structural support (sub-parts, marks)
  Poor (ambiguous) → Clarity and alignment improvements
"""

import re
from question_classifier import classify_question


class AssessmentImprovementAgent:
    def __init__(self):
        self.complexity_patterns = [
            (r"^(What|When|Where|Who) is (.+)\?$", r"Explain the significance of \2 and its impact."),
            (r"^Define (.+)\.$", r"Compare and contrast \1 with a related concept."),
            (r"^List (.+)\.$", r"Analyze the key components of \1."),
        ]

    def improve(self, question: str, issues: str, principles: str,
                stats: dict = None) -> str:
        """
        Propose improvements based on the question's classification.

        For recall-suitable questions, suggestions are framed as OPTIONAL
        upgrades, not corrections. The original question is not implied
        to be incorrect.
        """
        suggestions = []
        rewritten = question

        case = None
        if stats:
            case = classify_question(
                stats.get('avg_score', 50),
                stats.get('pass_rate', 50),
                stats.get('variance', 0)
            )

        if case == "easy":
            # ── Recall-suitable: optional upgrade suggestions ──
            suggestions.append(
                "This question is well-suited for recall assessment. "
                "No changes are required if the goal is a knowledge check."
            )
            suggestions.append(
                "Optional upgrade: If the goal is to assess higher-order "
                "thinking, consider rephrasing to require explanation or analysis."
            )

            for pattern, template in self.complexity_patterns:
                if re.match(pattern, question, re.IGNORECASE):
                    rewritten = re.sub(pattern, template, question, flags=re.IGNORECASE)
                    break
            else:
                if "?" in question:
                    rewritten = question.replace("?", "") + " and explain why?"
                else:
                    rewritten = question + " Explain your reasoning."

        elif case == "hard_valid":
            # ── Good discriminator: structural support ──
            suggestions.append(
                "The question effectively discriminates student ability. "
                "Consider adding structural support to help students "
                "demonstrate partial mastery."
            )
            suggestions.append(
                "Break the question into sub-parts with explicit marks "
                "for each step (e.g., part a, b, c)."
            )
            suggestions.append(
                "State all assumptions and given values explicitly in "
                "the question stem."
            )
            rewritten = (
                f"Answer the following in parts:\n"
                f"(a) [Setup/Definition step]\n"
                f"(b) [Core computation/reasoning step]\n"
                f"(c) {question}\n"
                f"Show all working. Marks are allocated per sub-part."
            )

        else:
            # ── Poor discriminator: clarity improvements ──
            if "Negative discrimination" in issues:
                suggestions.append(
                    "Review the answer key for correctness — high-performing "
                    "students may be selecting a response that appears correct "
                    "but is keyed differently."
                )
                suggestions.append(
                    "Clarify the distinction between the intended answer "
                    "and plausible distractors."
                )
                rewritten = f"Clarify specific constraints: {question}"
            else:
                suggestions.append(
                    "The question may benefit from clearer wording to improve "
                    "its discriminatory power."
                )
                suggestions.append(
                    "Consider aligning the question more closely with specific "
                    "learning outcomes to ensure it measures the intended competency."
                )
                rewritten = question.replace("?", "") + ". Explain your reasoning."

        # ── Format output ──
        output_lines = ["Suggested Improvements:"]
        for i, s in enumerate(suggestions, 1):
            output_lines.append(f"{i}. {s}")
        output_lines.append("")
        output_lines.append("Rewritten Question:")
        output_lines.append(rewritten)

        return "\n".join(output_lines)


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    agent = AssessmentImprovementAgent()

    test_cases = [
        {
            "label": "Easy recall",
            "q": "What is photosynthesis?",
            "issues": "Expected for recall-level questions...",
            "principles": "Formative assessment...",
            "stats": {"avg_score": 92, "pass_rate": 95, "variance": 30}
        },
        {
            "label": "Good discriminator",
            "q": "Derive the closed-form eigenvalue solution.",
            "issues": "High variance... positive discrimination...",
            "principles": "Scaffolding...",
            "stats": {"avg_score": 35, "pass_rate": 33, "variance": 1450}
        },
        {
            "label": "Poor discriminator",
            "q": "Which option best describes the process?",
            "issues": "Low variance with scores clustering...",
            "principles": "Clarity, Alignment...",
            "stats": {"avg_score": 52, "pass_rate": 50, "variance": 3}
        },
    ]

    print("Assessment Improvement Agent — Demo\n" + "=" * 50)
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Case {i}] {case['label']}")
        print(f"  Q: {case['q']}")
        print("-" * 30)
        print(agent.improve(case["q"], case["issues"], case["principles"], case["stats"]))
        print("=" * 50)
