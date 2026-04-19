"""
Pedagogical Knowledge Base
============================
Curated knowledge documents covering assessment design best practices,
Bloom's taxonomy, MCQ design principles, common question flaws,
difficulty calibration, and assessment fairness.

These documents serve as the retrieval corpus for the RAG pipeline.
Each document is a self-contained knowledge chunk optimized for
embedding and retrieval.
"""

PEDAGOGY_DOCUMENTS = [
    # ── Bloom's Taxonomy ──
    """Bloom's Taxonomy — Cognitive Domain Levels for Assessment Design

Bloom's Taxonomy defines six hierarchical levels of cognitive complexity that should guide assessment question design:

1. REMEMBER (Knowledge): Recall facts, terms, basic concepts. Questions at this level ask students to define, list, name, or identify. Example: "What is the chemical formula for water?" These are the easiest questions and should comprise only a portion of any assessment.

2. UNDERSTAND (Comprehension): Demonstrate understanding of facts and ideas. Questions ask students to explain, summarize, paraphrase, or interpret. Example: "Explain why water is essential for cellular processes." Slightly more complex than recall.

3. APPLY (Application): Use information in new situations. Questions ask students to solve, demonstrate, or implement. Example: "Calculate the pH of a 0.1M HCl solution." Requires transfer of knowledge to novel contexts.

4. ANALYZE (Analysis): Break information into parts, find patterns and relationships. Questions ask students to compare, contrast, differentiate, or examine. Example: "Compare the effectiveness of aerobic vs. anaerobic respiration." These questions effectively discriminate between strong and weak students.

5. EVALUATE (Evaluation): Justify decisions, critique, make judgments. Questions ask students to assess, defend, judge, or critique. Example: "Evaluate the ethical implications of genetic engineering in agriculture." High-order thinking.

6. CREATE (Synthesis): Produce new work, design solutions, construct arguments. Questions ask students to design, construct, develop, or formulate. Example: "Design an experiment to test the effect of pH on enzyme activity." The highest cognitive level.

Best Practice: A well-designed assessment includes questions across multiple Bloom's levels. Overreliance on Remember-level questions (which often produce high pass rates and low discrimination) fails to assess deeper understanding.""",

    """Bloom's Taxonomy and Difficulty Calibration

The relationship between Bloom's Taxonomy levels and question difficulty is not strictly linear but follows general patterns:

- Remember/Understand questions tend to be EASY (high pass rates, low variance among students)
- Apply/Analyze questions tend to be MEDIUM (moderate pass rates, higher variance)
- Evaluate/Create questions tend to be HARD (lower pass rates, highest variance)

When a question is classified as "Easy" with very high pass rates (>90%), consider whether it only tests recall. If so, revise it to target Apply or Analyze levels to improve its diagnostic value.

When a question is classified as "Hard" with very low pass rates (<20%), evaluate whether it targets an appropriate Bloom's level for the course. If it requires Create-level thinking in an introductory course, it may be unfairly difficult.

IMPORTANT: Difficulty should match learning objectives. A question isn't "bad" just because it's easy — if the objective is foundational knowledge, a Remember-level question is appropriate. The issue arises when ALL questions test the same cognitive level.""",

    # ── MCQ Design Best Practices ──
    """Multiple-Choice Question Design — Best Practices

A well-constructed MCQ consists of a STEM (the question or incomplete statement) and OPTIONS (one correct answer + distractors).

STEM GUIDELINES:
1. The stem should present a single, clear problem or question
2. Include as much of the question content in the stem as possible, not in the options
3. Avoid negative phrasing ("Which is NOT...") — if used, emphasize the negative word
4. Avoid verbose or unnecessarily complex language
5. The stem should make sense on its own without reading the options

OPTION GUIDELINES:
1. All options should be plausible and grammatically consistent with the stem
2. Options should be similar in length — a notably longer correct answer is a guessing cue
3. Avoid "All of the above" and "None of the above" — these reduce discrimination
4. Use 3-5 options (4 is optimal; research shows diminishing returns beyond 4)
5. Each distractor should represent a common student misconception
6. Arrange options in a logical order (numerical, alphabetical, or conceptual)

CORRECT ANSWER:
1. The correct answer should be unambiguously correct
2. It should not be distinguishable from distractors by non-content cues
3. Vary the position of the correct answer randomly across questions""",

    """Distractor Quality in Multiple-Choice Questions

High-quality distractors are the hallmark of effective MCQs. Poor distractors reduce the question's ability to discriminate between knowledgeable and unknowledgeable students.

SIGNS OF POOR DISTRACTORS:
- Distractors that no student selects (non-functional distractors)
- Distractors that are obviously wrong or absurd
- Distractors with grammatical inconsistencies with the stem
- Distractors that use absolute terms ("always," "never," "all," "none")
- Distractors significantly shorter or longer than the correct answer

IMPROVING DISTRACTORS:
1. Base distractors on common student errors and misconceptions
2. Use results from open-ended questions to identify typical wrong answers
3. Each distractor should be partially correct or represent a plausible reasoning error
4. Avoid overly similar distractors that confuse rather than assess
5. Test distractors empirically — if fewer than 5% of students select a distractor, replace it

DISCRIMINATION AND DISTRACTOR QUALITY:
- A question with good distractors will have a discrimination index > 0.3
- Poor distractors lead to low discrimination because guessing is easier
- The discrimination index measures how well the item separates high and low performers""",

    # ── Common Question Flaws ──
    """Common Flaws in Assessment Questions

AMBIGUITY:
- Questions with multiple valid interpretations lead to unfair assessment
- Avoid pronouns with unclear referents
- Avoid terms that can mean different things in different contexts
- Ensure there is exactly ONE unambiguously correct answer
- Have colleagues review questions for alternative interpretations

GRAMMATICAL CUES:
- When the stem uses "a/an," ensure all options match grammatically
- Verb tense agreement between stem and options
- Singular/plural agreement can inadvertently indicate the correct answer
- These cues allow test-wise students to guess correctly without knowledge

CONVERGENCE CUES:
- When three options say one thing and one says the opposite, students pick the majority
- When one option is a subset of another, students avoid it
- Overlapping options confuse students and reduce measurement quality

SPECIFICITY CUES:
- Overly specific correct answers stand out
- Options with qualifications ("usually," "sometimes") tend to be correct
- Options with absolutes ("always," "never") tend to be incorrect
- Savvy students exploit these patterns

ITEM WRITING FLAWS THAT INCREASE DIFFICULTY ARTIFICIALLY:
- Double negatives in the stem
- Complex sentence structures unrelated to the content
- Tricky wording designed to catch students rather than assess knowledge
- These create construct-irrelevant difficulty — measuring test-taking ability, not knowledge""",

    """Assessment Question Clarity Principles

CLARITY is the single most important factor in question quality. An unclear question measures reading comprehension and interpretation rather than content knowledge.

PRINCIPLES FOR CLEAR QUESTIONS:
1. Use simple, direct language appropriate to the students' level
2. Define technical terms if they are not part of the learning objective
3. Avoid jargon, colloquialisms, and culturally specific references
4. Each question should test a single concept or skill
5. State the question completely in the stem — don't rely on options for context
6. Use positive phrasing whenever possible
7. Review questions for reading level — questions should not be harder to READ than to ANSWER

READABILITY AND FAIRNESS:
- Non-native speakers and students with learning disabilities are disproportionately affected by poor clarity
- Unnecessarily complex language introduces construct-irrelevant variance
- If variance in scores is high, consider whether the question's complexity is in the content or the wording

PEER REVIEW PROTOCOL:
- Have at least one colleague answer each question independently
- Ask: "Is there only one correct answer?" "Is the question clear?" "Are distractors plausible?"
- Revise based on feedback before deploying in a live assessment""",

    # ── Difficulty Calibration ──
    """Difficulty Calibration Strategies for Assessment Design

OPTIMAL DIFFICULTY RANGE:
- For norm-referenced assessments: target 40-60% average scores (medium difficulty)
- For criterion-referenced assessments: difficulty should match the learning objective
- A well-balanced assessment includes: ~30% easy, ~40% medium, ~30% hard questions
- This distribution maximizes information about student ability across the full range

WHEN QUESTIONS ARE TOO EASY (avg > 80%, pass rate > 90%):
1. Move up one Bloom's Taxonomy level
2. Add complexity by requiring application of multiple concepts
3. Improve distractors to represent more plausible alternatives
4. Add a context or scenario that requires transfer of knowledge
5. Do NOT make questions harder by adding trick elements or ambiguity

WHEN QUESTIONS ARE TOO HARD (avg < 30%, pass rate < 20%):
1. Check that the content was adequately covered in instruction
2. Simplify language without simplifying the concept being tested
3. Reduce the number of steps required to reach the answer
4. Provide more context or scaffolding in the stem
5. Ensure distractors are clearly wrong (not confusingly similar to the correct answer)
6. Consider whether the question tests prerequisite knowledge from a previous course

DIFFICULTY VS. DISCRIMINATION:
- Easy questions (>90% correct) have low discrimination — everyone gets them right
- Very hard questions (<10% correct) also have low discrimination — everyone gets them wrong
- Medium-difficulty questions (40-70% correct) tend to have the HIGHEST discrimination
- For formative assessment, easier questions are acceptable; for summative, prioritize discrimination""",

    # ── Assessment Fairness ──
    """Assessment Fairness and Bias Avoidance

PRINCIPLES OF FAIR ASSESSMENT:
1. Questions should measure knowledge/skills, not cultural background
2. Avoid scenarios that advantage students from specific demographic groups
3. Ensure diverse representation in question contexts
4. Questions should not require knowledge outside the curriculum
5. Time limits should be sufficient for all students, including non-native speakers

DETECTING BIAS IN ITEM STATISTICS:
- Differential Item Functioning (DIF) analysis compares performance across demographic groups
- High variance combined with moderate average scores may indicate bias
- If specific demographic groups consistently underperform on a question with similar overall ability, the question may be biased

TYPES OF ASSESSMENT BIAS:
1. CONTENT BIAS: Question content is more familiar to some groups
2. LINGUISTIC BIAS: Language complexity disadvantages non-native speakers
3. CULTURAL BIAS: Scenarios assume cultural knowledge not shared by all students
4. FORMAT BIAS: Question format favors certain learning styles

STATISTICAL INDICATORS OF POTENTIAL UNFAIRNESS:
- Very high variance (>250) may indicate the question is differentially accessible
- Low discrimination combined with moderate difficulty suggests the question may not be measuring the intended construct
- Unusual patterns in distractor selection may indicate cultural differences in interpretation

RESPONSIBLE AI IN ASSESSMENT:
- Automated assessment analysis tools provide decision SUPPORT, not decisions
- Human judgment should always be the final arbiter of question quality
- Statistical indicators should be interpreted in context of the specific student population
- Transparent reporting of model limitations is an ethical requirement""",

    # ── Item Analysis and Discrimination ──
    """Item Analysis — Statistical Quality Indicators

DISCRIMINATION INDEX:
The discrimination index measures how well a question differentiates between high-performing and low-performing students.

Calculation: D = (P_upper - P_lower) where P_upper is the proportion of top-27% students who answered correctly and P_lower is the proportion of bottom-27% students who answered correctly.

Interpretation:
- D > 0.40: Excellent discrimination — keep this question
- D = 0.30–0.39: Good discrimination — minor revisions may help
- D = 0.20–0.29: Marginal — question needs revision
- D < 0.20: Poor — question should be significantly revised or removed
- D < 0.00: Negative — this means low performers did BETTER than high performers, which indicates a seriously flawed question

LOW DISCRIMINATION CAUSES:
1. Ambiguous question that confuses knowledgeable students
2. Non-functional distractors that don't attract any students
3. Question tests trivial recall that everyone knows
4. Question is so hard that everyone guesses randomly
5. Correct answer has grammatical or formatting cues

POINT-BISERIAL CORRELATION:
A more sophisticated measure that correlates the item score with total test score.
- Values > 0.30 indicate good item quality
- Values < 0.10 indicate the item doesn't contribute meaningfully to the assessment

DISTRACTOR ANALYSIS:
For each distractor:
- Calculate selection rate: each distractor should attract at least 5% of students
- Compare selection by upper vs. lower groups: distractors should attract MORE low performers
- Non-functional distractors (selected by <5%) should be replaced""",

    """Student Performance Patterns and Assessment Interpretation

INTERPRETING AVERAGE SCORES:
- Average > 80%: Content is well-mastered OR question is too easy
- Average 60-80%: Good range for most assessment purposes
- Average 40-60%: Challenging but fair for advanced content
- Average < 40%: Potential issues — review instruction and question quality

INTERPRETING VARIANCE:
- Low variance (<50): Students performed uniformly — question doesn't discriminate
- Moderate variance (50-200): Healthy spread indicating the question differentiates
- High variance (>200): Significant split in the class — investigate potential causes
- Very high variance (>300): Red flag — possible bias, prerequisite issues, or ambiguity

INTERPRETING PASS RATE:
- Pass rate > 90%: Consider if question is appropriately challenging
- Pass rate 60-90%: Good range for formative assessment
- Pass rate 40-60%: Appropriate for summative assessment of complex content
- Pass rate < 40%: Investigate — is the question fair? Was content taught?

COMBINED PATTERN ANALYSIS:
- Low average + Low variance = Uniformly poor performance → Instruction or question issue
- Low average + High variance = Some students get it, many don't → Prerequisite gap
- High average + Low variance = Uniformly good performance → Question may be too easy
- High average + High variance = Most do well but some significantly struggle → Individual support needed
- Medium average + High variance = Classic difficulty calibration → Good assessment item

THE ROLE OF SIMULATED VS. REAL SCORES:
When using simulated scores for model training, the relationship between text features and performance features is synthetic. During live inference with real user-provided scores, the model leverages actual student outcomes to make more authentic predictions. The model's prediction quality during live use depends on the quality and representativeness of the student score data provided.""",
]
