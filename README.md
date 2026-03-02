# Project: Intelligent Exam Question Analysis & Agentic Pedagogical Strategy

## From Predictive Analytics to Intelligent Intervention

### Project Overview
This project involves the design and implementation of an **AI-driven educational analytics system** that predicts exam question difficulty and evolves into an agentic AI pedagogical assistant.

- **Milestone 1:** Classical machine learning techniques applied to exam question text and simulated student performance behavior data to predict question difficulty (Easy / Medium / Hard) and identify key performance drivers.
- **Milestone 2:** Extension into an agent-based AI application that autonomously reasons about difficulty factors, retrieves educational best practices (RAG), and plans improvement strategies for educators.

---

### Constraints & Requirements
- **Team Size:** 3–4 Students
- **API Budget:** Free Tier Only (Open-source models / Free APIs)
- **Framework:** LangGraph (Recommended)
- **Hosting:** Mandatory (Hugging Face Spaces, Streamlit Cloud, or Render)

---

### Technology Stack
| Component | Technology |
| :--- | :--- |
| **ML Models (M1)** | Logistic Regression, Decision Trees, Scikit-Learn |
| **Agent Framework (M2)** | LangGraph, Chroma/FAISS (RAG) |
| **UI Framework** | Streamlit or Gradio |
| **LLMs (M2)** | Open-source models or Free-tier APIs |

---

### Milestones & Deliverables

#### Milestone 1: ML-Based Difficulty Prediction (Mid-Sem)
**Objective:** Identify exam question difficulty using text features along with historical student behavioral data, focusing strictly on classical ML pipelines *without LLMs*.

**Key Deliverables:**
- Problem understanding & Business context.
- System architecture diagram.
- Working local application with UI (Streamlit/Gradio).
- Model performance evaluation report (Accuracy, Precision, Recall, Confusion Matrix).

#### Milestone 2: Agentic Pedagogical Assistant (End-Sem)
**Objective:** Extend the system into an agentic strategist that reasons about linguistic complexity and performance variance, retrieving best pedagogical practices to generate structured recommendations.

**Key Deliverables:**
- **Publicly deployed application** (Link required).
- Agent workflow documentation (States & Nodes).
- Structured pedagogical report generation.
- GitHub Repository & Complete Codebase.
- Demo Video (Max 5 mins).

---

### Evaluation Criteria

| Phase | Weight | Criteria |
| :--- | :--- | :--- |
| **Mid-Sem** | 25% | ML technique application, Feature Engineering (TF-IDF & scaling), UI Usability, Evaluation Metrics. |
| **End-Sem** | 30% | Reasoning quality, RAG & State management implementation, Output clarity, Deployment success. |

> [!WARNING]
> Localhost-only demonstrations will **not** be accepted for final submission. Project must be hosted.
