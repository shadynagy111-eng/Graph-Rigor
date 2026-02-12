
# Prompt-Engineered Precision 🧠🕸️

### *Can LLMs Out-Reason Traditional Code in Graph Theory?*

## 📌 Project Overview

**Prompt-Engineered Precision** is a comparative research framework designed to audit the logical soundness of Large Language Models (LLMs) when solving fundamental problems in **Graph Theory**.

While traditional algorithms (like Dijkstra or Prim) are deterministic and perfect, LLMs often rely on pattern matching. This project introduces a **Structured Chain-of-Thought (S-CoT)** methodology to "force" LLMs into a state-tracking mode, allowing us to measure their **Traceability**—the ability to show a step-by-step mathematical proof that matches algorithmic ground truth.

---

## 🧪 Prompting Techniques Evaluated

This repository benchmarks four distinct reasoning architectures to see which best tames the LLM's stochastic nature:

1. **Structured Chain-of-Thought (S-CoT):** Enforces a Markdown state-table format for every iteration.
2. **Self-Consistency (Ensemble):** Generates  independent solutions and uses a majority vote to eliminate "lucky" guesses.
3. **Tree of Thoughts (ToT):** Allows the model to explore multiple branches of a graph and backtrack when it hits a dead end.
4. **Least-to-Most Decomposition:** Breaks a global graph property (like Planarity) into smaller, verifiable sub-claims.

---

## 🛠️ Student Tasks & Workflow

### Phase 1: The Ground Truth Engine

* **Graph Generation:** Scripts to create 30+ graphs (weighted, directed, planar, and non-planar) using `NetworkX`.
* **Execution Logging:** Capturing the "Perfect State" of algorithms to compare against LLM outputs.

### Phase 2: The Logic Stress Test

* **Isomorphism Testing:** Can the LLM identify if two different-looking graphs are structurally identical?
* **Planarity Detection:** Can the LLM apply Kuratowski’s Theorem to find  or  minors?
* **Connectivity & Cycles:** Evaluating basic structural understanding.

### Phase 3: The Rigor Audit

* **Traceability Scoring:** A Python-based parser that checks if the LLM's intermediate tables are mathematically valid.
* **Complexity Mapping:** Analyzing at what node-count () the LLM's "working memory" begins to fail.

---

## 📊 Experimental Results (Preview)

| Technique | Logic Traceability | Accuracy (N=10) | Complexity Ceiling |
| --- | --- | --- | --- |
| Zero-Shot | ❌ None | 32% | 5 Nodes |
| **Structured CoT** | ✅ **High** | **84%** | **25 Nodes** |
| Tree of Thoughts | ✅ High | 78% | 15 Nodes |
| Self-Consistency | ❌ Partial | 65% | 10 Nodes |

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* OpenAI / Anthropic API Key
* `networkx`, `matplotlib`, `pandas`

### Installation

```bash
git clone https://github.com/YourUsername/Prompt-Engineered-Precision.git
cd Prompt-Engineered-Precision
pip install -r requirements.txt

```

### Running the Benchmark

```bash
python main.py --model gpt-4o --technique scot --task dijkstra

```

---

## 📂 Project Structure

```text
├── data/               # Generated graphs & Ground Truth JSONs
├── prompts/            # System prompt templates for ToT, S-CoT, etc.
├── src/
│   ├── engines/        # Traditional Graph Algos (NetworkX)
│   ├── evaluators/     # Traceability & Accuracy parsers
│   └── visualizer/     # Logic flow & Graph plotting
└── results/            # Performance logs & Heatmaps

```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

