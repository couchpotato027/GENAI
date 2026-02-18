"""
Pedagogical Retrieval Agent
===========================
Retrieves relevant pedagogical principles from a knowledge base
based on the question's psychometric classification.

Case-driven retrieval:
  Easy (recall)   → formative assessment, knowledge checks
  Hard (valid)    → scaffolding, partial credit, stepwise marking
  Poor (ambiguous)→ clarity, alignment, ambiguity removal
"""

import os
import re
from question_classifier import classify_question


class PedagogicalRetrievalAgent:
    def __init__(self, kb_path: str = "pedagogy_kb.md"):
        self.kb_path = kb_path
        self.kb_content = self._load_kb()

    def _load_kb(self) -> str:
        """Load the knowledge base content."""
        if not os.path.exists(self.kb_path):
            return "Knowledge base not found."
        with open(self.kb_path, 'r') as f:
            return f.read()

    def retrieve_principles(self, issues: str, stats: dict = None) -> str:
        """
        Retrieve relevant pedagogical principles based on the question's
        psychometric classification and identified observations.

        Parameters
        ----------
        issues : String containing observations from AssessmentAnalysisAgent.
        stats  : Dict with 'avg_score', 'pass_rate', 'variance'.

        Returns
        -------
        Formatted string with Retrieved Pedagogical Principles.
        """
        retrieved = []

        case = None
        if stats:
            case = classify_question(
                stats.get('avg_score', 50),
                stats.get('pass_rate', 50),
                stats.get('variance', 0)
            )

        if case == "easy":
            # Recall-suitable: retrieve formative assessment principles
            section = self._extract_section("### Formative Assessment & Recall")
            if section:
                retrieved.append(section.strip())

        elif case == "hard_valid":
            # Good discriminator: retrieve scaffolding/structure principles
            section = self._extract_section("### Hard but Well-Designed")
            if section:
                retrieved.append(section.strip())

        else:
            # Poor discriminator: retrieve clarity/alignment principles
            mapping = {
                "Negative discrimination": "### Negative Discrimination",
                "Low variance": "### Low Discrimination",
                "Content mismatch": "## Clarity & Alignment",
            }
            for keyword, section_header in mapping.items():
                if keyword in issues:
                    section = self._extract_section(section_header)
                    if section:
                        retrieved.append(section.strip())

        # Fallback
        if not retrieved:
            fallback_mapping = {
                "Negative discrimination": "### Negative Discrimination",
                "Low discrimination": "### Low Discrimination",
                "ceiling effect": "### Extremely Low Difficulty",
                "Content mismatch": "## Clarity & Alignment",
            }
            for keyword, section_header in fallback_mapping.items():
                if keyword.lower() in issues.lower():
                    section = self._extract_section(section_header)
                    if section:
                        retrieved.append(section.strip())

        if not retrieved:
            return "Retrieved Pedagogical Principles:\n- No specific principles required — the question is well-suited for its intended assessment purpose."

        return "Retrieved Pedagogical Principles:\n" + "\n\n".join(retrieved)

    def _extract_section(self, header: str) -> str:
        """Extract the content of a specific markdown section."""
        if header not in self.kb_content:
            return ""
        start_idx = self.kb_content.find(header) + len(header)
        rest_of_text = self.kb_content[start_idx:]
        next_header_match = re.search(r'\n#+\s', rest_of_text)
        content = rest_of_text[:next_header_match.start()] if next_header_match else rest_of_text
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    agent = PedagogicalRetrievalAgent()

    test_cases = [
        {
            "label": "Easy recall question",
            "issues": "High pass rate... expected for recall-level questions...",
            "stats": {"avg_score": 92, "pass_rate": 95, "variance": 30}
        },
        {
            "label": "Hard, good discriminator",
            "issues": "High variance... positive discrimination...",
            "stats": {"avg_score": 35, "pass_rate": 33, "variance": 1450}
        },
        {
            "label": "Ambiguous, poor discriminator",
            "issues": "Low variance with scores clustering near the mean...",
            "stats": {"avg_score": 52, "pass_rate": 50, "variance": 3}
        },
    ]

    print("Pedagogical Retrieval Agent — Demo\n" + "=" * 50)
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Case {i}] {case['label']}")
        print("-" * 30)
        print(agent.retrieve_principles(case["issues"], case["stats"]))
        print("=" * 50)
