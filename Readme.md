# SVM - Bundle Method

This project implements Support Vector Machine (SVM) training using **bundle methods** applied to the **primal, non-smooth formulation** of SVMs with hinge loss. The focus is on solving large-scale optimization problems using a convex but non-differentiable objective.

---

## 📌 Key Features

- Primal SVM optimization using **hinge loss** (no slack variables)
- Support for **explicit polynomial kernels**
- Optimization via **proximal bundle method**
- Modular code structure:
  - Datasets or (generation & loading)
  - Experiments with varying scale and feature complexity
  - Reporting and plotting tools

---

## 📁 Project Structure

SVM - BUNDLE METHOD/ 
│ 
├── Datasets/ # Synthetic or real data generation/loading 
├── Experiments/ # Scripts for running bundle method experiments 
├── Functions/ # Core implementations: SVM model, bundle method logic 
├── Notes\ Guide/ # Notes, derivations, and theoretical documentation 
├── Plots/ # Output visualizations (e.g. convergence, timing) 
├── Reports/ # Final reports and paper drafts 
├── .gitignore # Ignore logs, outputs, checkpoints, etc. 
└── Readme.md # Project overview and instructions