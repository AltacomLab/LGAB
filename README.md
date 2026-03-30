# LGAB: Learning-Guided Approximate Bisimulation

This repository contains the implementation of **Learning-Guided Approximate Bisimulation (LGAB)** for scalable reasoning in quantitative transition systems (QTS) and knowledge graphs (KGs), as described in:

Learning guided approximate reasoning in quantitative transition systems. 

## Features
- Hybrid neuro-symbolic framework combining neural approximation and semantic refinement
- Scalable bisimulation computation with near-quadratic complexity
- Optional LLM-guided policy for critical state prioritization
- Supports large-scale time-series and knowledge graph datasets

## Repository Structure
LGAB/
├── README.md
├── LICENSE
├── .gitignore
├── data/
│ ├── electricity/
│ ├── ett/
│ └── renewable_energy_kg/
├── src/
│ ├── dataset/
│ ├── models/
│ ├── training/
│ ├── evaluation/
│ └── utils/
├── experiments/
│ ├── configs/
│ └── scripts/
├── notebooks/
├── results/
└── requirements.txt
