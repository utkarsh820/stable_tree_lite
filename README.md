# Stable Decision Trees (Lite Research Implementation)

This is a minimal, research-grade prototype inspired by the paper:  
**"Improving Stability in Decision Tree Models"**  
[arXiv:2305.17299](https://arxiv.org/pdf/2305.17299) – MIT CSAIL

---

##  Motivation

Decision trees are powerful yet notoriously unstable models — small perturbations in the training data can lead to drastically different trees. This project explores a technique for making decision trees **structurally stable**, using a form of regularization that penalizes divergence from a "reference" tree during training.

---

##  Core Idea

We introduce a custom class `StableDecisionTree`, where training includes a **stability regularization** term:

\[
\text{Total Loss} = \text{Classification Loss} + \lambda \times \text{TreeDistance}
\]

- **TreeDistance** is a structural distance metric between two trees (original vs. perturbed).
- **λ (lambda)** controls the trade-off between accuracy and stability.
- The goal: produce trees that retain similar structure under small data perturbations.

---

##  Tree Distance Metric (Out-Distance)

We use a simplified version of the **"out-distance"** proposed in the paper:

- Measures which features are used in the tree
- Considers feature **depth** and **position** in the structure
- Returns a normalized distance ∈ [0, 1], where:
  - 0 → identical trees  
  - 1 → completely different trees

This allows quantifying structural divergence under perturbation.

---

##  Features

-  Lightweight and readable `StableDecisionTree` implementation  
-  Structural comparison via `tree_distance()`  
-  Data perturbation and re-training for empirical evaluation  
-  Lambda-sweep experiment for analyzing the **accuracy–stability trade-off**  
-  Visual and tabular comparison of decision tree structures  

---

##  Results Snapshot

index lambda  mean_distance  mean_accuracy
0     0.0          0.547          0.942
1     0.1          0.574          0.942
2     0.5          0.556          0.942
3     1.0          0.539          0.942
4     2.0          0.580          0.942
5     5.0          0.522          0.942

- As λ increases, tree structure becomes more **stable**, while accuracy decreases slightly.
- Sweet spot depends on application — particularly useful when **interpretability** and **robustness** matter.

---

##  Use of LLMs

LLMs (e.g., ChatGPT, LLM Notebook) were used minimally and transparently for:

- Explaining dense sections of the research paper
- Prototyping markdown documentation
- Isolated code refactoring suggestions
- code generation (help / idea)

All core modeling, experimentation, and interpretation were independently implemented and verified.

---

##  Project Structure

stable-tree-lite/
│
├── data/                      # CSVs, small UCI datasets
│
├── core/                      # Clean but simple code
│   ├── stable_tree.py         # StableDecisionTree class
│   ├── tree_distance.py       # Structural distance function
│   └── perturbation.py        # Optional: dataset perturbations
│
├── results/                   # Store plots, metrics, etc.
│
├── notebooks/
│   └── 01_stable_tree_experiments.ipynb   # Single, full exploratory notebook
│
├── requirements.txt
├── README.md
└── .gitignore

##  Reference

> *Improving Stability in Decision Tree Models*  
> [arXiv:2305.17299](https://arxiv.org/pdf/2305.17299)  
> Authors: MIT CSAIL  
> Summary: Proposes a regularized objective for training stable decision trees under dataset perturbations, leveraging out-distance as a structural penalty.

##  Feedback

Pull requests, issues, and experimental forks are welcome — this is a research-first, community-friendly project!