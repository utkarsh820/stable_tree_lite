# Stable Decision Trees (Lite Research Implementation)

> **Inspired by the paper:**  
> [Improving Stability in Decision Tree Models](https://arxiv.org/pdf/2305.17299) — MIT CSAIL

---

##  Motivation

Decision trees are powerful yet notoriously **unstable** models — small perturbations in training data can lead to **drastically different trees**. This project explores a technique to make decision trees **structurally stable**, by adding a regularization term that penalizes divergence from a "reference" tree.

---

##  Core Idea

We introduce a custom class `StableDecisionTree`, where training includes a **stability regularization** term:

\[
\text{Total Loss} = \text{Classification Loss} + \lambda \times \text{TreeDistance}
\]

- `TreeDistance`: a structural distance between two trees (original vs. perturbed)
- `λ (lambda)`: controls the trade-off between accuracy and stability
- **Goal**: produce trees that remain structurally similar under slight data changes

---

##  Tree Distance Metric (Out-Distance)

We use a simplified form of **out-distance** from the original paper:

- Measures **which features** are used
- Considers **depth** and **position** of splits
- Returns a normalized distance in **\[0, 1\]**:
  - `0` → identical structure  
  - `1` → completely different structure

This quantifies **structural divergence** due to perturbation.

---

##  Features

- Lightweight `StableDecisionTree` implementation  
- Structural comparison with `tree_distance()`  
- Dataset perturbation and re-training routines  
- Lambda sweep to study **accuracy–stability trade-off**  
- Visual and tabular comparisons of decision trees  

---

##  Results Snapshot

| Index | Lambda | Mean Distance | Mean Accuracy |
|-------|--------|----------------|----------------|
| 0     | 0.0    | 0.547          | 0.942          |
| 1     | 0.1    | 0.574          | 0.942          |
| 2     | 0.5    | 0.556          | 0.942          |
| 3     | 1.0    | 0.539          | 0.942          |
| 4     | 2.0    | 0.580          | 0.942          |
| 5     | 5.0    | 0.522          | 0.942          |

- As `λ` increases, **tree structure stabilizes**, with only minor loss in accuracy.
- Sweet spot depends on your **use case** — this is ideal when **interpretability** and **robustness** are key.

---

##  Use of LLMs

LLMs (e.g., ChatGPT, LLM Notebook) were used **minimally and transparently** for:

- Clarifying complex parts of the original research paper
- Prototyping markdown and documentation
- Refactoring ideas and isolated code snippets

**All core modeling and experimentation** were independently implemented and verified.

---

## 🗂 Project Structure

```

stable-tree-lite/
│
├── data/                      # CSVs, small UCI datasets
│
├── core/                      # Clean but simple code
│   ├── stable\_tree.py         # StableDecisionTree class
│   ├── tree\_distance.py       # Structural distance function
│   └── perturbation.py        # Optional: dataset perturbations
│
├── results/                   # Store plots, metrics, etc.
│
├── notebooks/
│   └── 01\_stable\_tree\_experiments.ipynb   # Main exploratory notebook
│
├── requirements.txt
├── README.md
└── .gitignore

```

---

##  Reference

> **Improving Stability in Decision Tree Models**  
> [arXiv:2305.17299](https://arxiv.org/pdf/2305.17299)  
> Authors: MIT CSAIL  
> Summary: Introduces a regularized objective for training stable decision trees using out-distance as a structural penalty term.

---

## 🤝 Feedback

Pull requests, issues, and experimental forks are welcome!  
This is a **research-first**, **community-friendly** project.
