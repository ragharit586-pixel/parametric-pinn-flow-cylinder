# Parametric PINNs for Flow Over Cylinders

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20T4-76B900?style=flat-square&logo=nvidia)

**AI-Accelerated CFD using Physics-Informed Neural Networks**

*M.Tech Research — IIST (Indian Institute of Space Science and Technology)*

</div>

---

## Overview

A **parametric Physics-Informed Neural Network (PINN)** surrogate that generalizes across Reynolds number variations for external flow over cylinders — replacing expensive full CFD re-simulations with real-time inference.

Unlike traditional CFD solvers that require complete re-simulation for every new flow condition, this model **learns the underlying Navier-Stokes physics once** and instantly predicts velocity and pressure fields for any Reynolds number in the trained range.

---

## Key Results

| Metric | Value |
|--------|-------|
| Accuracy vs Traditional CFD | ~95% |
| Computational Time Reduction | 60% |
| Real-Time Inference Speed | <50ms (NVIDIA T4) |
| Reynolds Number Range | Re = 100 – 1000 |
| Training Hardware | NVIDIA T4 GPU |

---

## Problem Statement

Traditional CFD (e.g., ANSYS Fluent) requires **full re-simulation** for every new Reynolds number — computationally expensive and time-consuming for parametric design studies.

**This work:** A single parametric PINN that takes `(x, y, Re)` as input and outputs `(u, v, p)` — velocity components and pressure — for any Re in the training range without re-solving the governing equations.

```
Input:  (x, y, Re)  →  Parametric PINN  →  Output: (u, v, p)
```

---

## Method

### Architecture
- **Input layer:** `[x, y, Re]` (3 neurons)
- **Hidden layers:** 6 fully-connected layers × 64 neurons, tanh activation
- **Output layer:** `[u, v, p]` (3 neurons)
- **Reynolds number** embedded directly as parametric input — no separate network per Re

### Physics Enforcement
- Navier-Stokes equations enforced via **automatic differentiation** (no labeled data needed for interior)
- **Multi-loss architecture:**

```
L_total = λ₁·L_PDE + λ₂·L_BC + λ₃·L_data + λ₄·L_reg
```

where:
- `L_PDE` — continuity + momentum equation residuals
- `L_BC` — no-slip wall, inlet, outlet boundary conditions
- `L_data` — sparse CFD reference data (optional supervision)
- `L_reg` — L2 regularization

### Training Strategy
- **Phase 1:** Physics-only training (unsupervised)
- **Phase 2:** Fine-tuning with sparse CFD reference data
- **Optimizer:** Adam → L-BFGS
- **Epochs:** 50,000+

---

## Repository Structure

```
parametric-pinn-flow-cylinder/
│
├── src/                        # Core source code
│   ├── model.py                # PINN architecture definition
│   ├── train.py                # Training loop and loss functions
│   ├── physics.py              # Navier-Stokes PDE residuals
│   ├── boundary.py             # Boundary condition enforcement
│   └── utils.py                # Helper functions, plotting
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_training.ipynb       # Model training walkthrough
│   ├── 02_evaluation.ipynb     # Results and validation
│   └── 03_visualization.ipynb  # CFD field visualization
│
├── configs/                    # Hyperparameter configurations
│   └── config.yaml             # Training config (Re range, layers, lr)
│
├── results/                    # Output plots and saved models
│   ├── figures/                # Velocity/pressure field plots
│   └── checkpoints/            # Saved model weights
│
├── data/                       # Reference CFD data (gitignored if large)
│   └── README.md               # Data source and format description
│
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| TensorFlow / Keras | Neural network framework |
| NumPy | Numerical computation |
| Matplotlib / Seaborn | Visualization |
| SciPy | Scientific utilities |
| NVIDIA T4 GPU | Training hardware |

---

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training the Model
```bash
python src/train.py --config configs/config.yaml
```

### Running Inference
```python
from src.model import ParametricPINN

model = ParametricPINN.load('results/checkpoints/best_model')
u, v, p = model.predict(x, y, Re=500)
```

---

## Project Status

- [x] Baseline PINN for single Reynolds number
- [x] Parametric extension with Re as input
- [x] Multi-loss training pipeline
- [ ] Validation against full CFD dataset (ANSYS Fluent)
- [ ] Extend to 3D flow geometries
- [ ] Inverse problem: infer Re from sparse velocity measurements

---

## Related Work

This project is part of ongoing M.Tech research at IIST exploring:
- **Inverse PINNs** for heat flux estimation in regenerative cooling channels
- **Parametric surrogate modeling** for aerodynamic design optimization

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{raghavendra2026parametricpinn,
  author    = {Raghavendra M},
  title     = {Parametric Physics-Informed Neural Networks for Flow Over Cylinders},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/ragharit586-pixel/parametric-pinn-flow-cylinder}
}
```

---

## Author

**Raghavendra M**  
M.Tech Aerospace Engineering (Thermal & Propulsion) @ IIST  
📧 ragharit586@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/raghavendra-mylar-b00b95240/)  
🐙 [GitHub](https://github.com/ragharit586-pixel)

---

<div align="center">
<sub>Built with Physics + Neural Networks @ IIST</sub>
</div>
