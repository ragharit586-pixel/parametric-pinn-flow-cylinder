# Parametric PINNs for Flow Over Cylinders

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Accuracy-97--98%25-success?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20T4-76B900?style=flat-square&logo=nvidia)

**Physics-Informed Neural Networks for CFD — No Flow Field Data Required**

*M.Tech Research — IIST (Indian Institute of Space Science and Technology)*

</div>

---

## Overview

A **parametric Physics-Informed Neural Network (PINN)** that learns steady-state flow over cylinders **without any CFD solution data**. The model generalizes across Reynolds numbers (Re = 10–47) using only the Navier-Stokes equations enforced via automatic differentiation, with optional benchmark-informed constraints for enhanced accuracy.

Unlike traditional CFD solvers that require expensive re-simulation for every new Reynolds number, this PINN **learns the underlying physics once** and predicts velocity and pressure fields for any Re in the trained range with real-time inference.

---

## Key Results

| Metric | Value |
|--------|-------|
| **Training Data** | **No CFD field data** (u,v,p) |
| **Physics-Only Accuracy** | 95–96% (4–5% error) |
| **Benchmark-Informed Accuracy** | **97–98% (1–2% error)** |
| **Real-Time Inference Speed** | <50ms (NVIDIA T4) |
| **Reynolds Number Range** | Re = 10 → 47 (steady regime) |
| **Training Hardware** | NVIDIA T4 GPU (Kaggle) |

---

## Problem Statement

Traditional CFD (e.g., ANSYS Fluent) requires **full re-simulation** for every new Reynolds number — computationally expensive and time-consuming for parametric design studies.

**This work:** A single parametric PINN that takes `(x, y, Re)` as input and outputs `(ψ, p)` — stream function and pressure — for any Re in the steady flow regime **without labeled CFD training data**.

```
Input:  (x, y, Re)  →  Parametric PINN  →  Output: (ψ, p)  →  Derive: (u, v)
```

### Why This Matters
- **No CFD solution fields required** — learns from physics equations only
- **Instant parametric predictions** — no re-simulation needed
- **Steady regime coverage** — Re = 10–47 (before vortex shedding onset)
- **Flexible training modes** — physics-only baseline or benchmark-informed refinement

---

## Method

### Architecture
- **Input layer:** `[x, y, Re]` (3 neurons) — spatial coordinates + Reynolds number
- **Hidden layers:** **8 fully-connected layers × 40 neurons**, tanh activation
- **Output layer:** `[ψ, p]` (2 neurons) — stream function + pressure
- **Input scaling:** All inputs normalized to [-1, 1] for balanced gradients
- **Velocity derivation:** `u = ∂ψ/∂y`, `v = -∂ψ/∂x` via automatic differentiation

### Reynolds Number Configuration

| Split | Re Values | Count |
|-------|-----------|-------|
| **Training** | 10, 15, 20, 25, 30, 35, 40 | 7 |
| **Interpolation test** | 12 (between 10–15), 28 (between 25–30) | 2 |
| **Extrapolation test** | 42, 45 (beyond training, still < Re_critical=47) | 2 |

All Reynolds numbers stay **below Re_critical = 47** to remain in the steady-state regime (no vortex shedding).

---

## Training Modes

This repository implements **two training approaches** to demonstrate both unsupervised physics learning and benchmark-informed refinement:

### Mode A: Physics-Only (Unsupervised)

Uses **only** Navier-Stokes PDE residuals and boundary conditions — zero external data:

```
L_total = L_PDE + β_BC · L_BC
```

- **Accuracy:** 95–96% (4–5% error)
- **Advantages:** Fully unsupervised, generalizes well
- **Limitations:** Slightly lower accuracy on integral quantities (Cd, wake length)

### Mode B: Physics + Benchmark-Informed Constraints (Weakly Supervised)

Adds weak supervision from **literature/benchmark scalars** (not CFD fields) to refine integral quantities:

```
L_total = L_PDE + β_BC·L_BC + β_Cd·L_Cd + β_wake·L_wake + β_Bern·L_Bernoulli + β_surf·L_surface
```

- **Accuracy:** **97–98% (1–2% error)**
- **Advantages:** Higher accuracy on Cd and wake physics
- **Note:** Uses only scalar targets from Dennis & Chang (1970) — **no CFD solution fields (u,v,p)**

---

## Multi-Objective Loss Function (Mode B)

**No CFD field data used.** The network learns from **6 physics-based loss terms**:

### Loss Weights

| Loss Term | Symbol | Weight (β) | Purpose |
|-----------|--------|------------|---------|
| PDE residuals | `L_PDE` | 1.0 | Navier-Stokes continuity + momentum |
| Boundary conditions | `L_BC` | **6.0** | Inlet, no-slip, outlet BCs |
| Drag coefficient | `L_Cd` | 0.4 | Match Dennis & Chang (1970) benchmark |
| Wake length | `L_wake` | 0.5 | Recirculation zone size constraint |
| Bernoulli regularization | `L_Bern` | 0.3 | Outer flow pressure-velocity consistency |
| Surface pressure | `L_surf` | 0.2 | Correct pressure gradient on cylinder |

### Governing Equations (PDE Loss)
- **Continuity:** `∂u/∂x + ∂v/∂y = 0`
- **Momentum-x:** `u·∂u/∂x + v·∂u/∂y + ∂p/∂x − ν·∇²u = 0`
- **Momentum-y:** `u·∂v/∂x + v·∂v/∂y + ∂p/∂y − ν·∇²v = 0`
- Kinematic viscosity `ν = ν(Re)` computed dynamically: `ν = U·D/Re`

### Boundary Conditions (BC Loss)
- **Inlet:** Parabolic velocity profile
- **Cylinder surface:** No-slip condition (`u = v = 0`)
- **Outlet:** Pressure outlet

### Wake Length Targets (from Literature)

| Re | Target Wake Length (×D) |
|----|------------------------|
| 10 | 1.45 |
| 15 | 1.65 |
| 20 | 1.95 |
| 25 | 2.25 |
| 30 | 2.60 |
| 35 | 3.05 |
| 40 | 3.50 |

---

## Training Strategy — 3-Phase Pipeline

### Phase 1: Adam Optimizer
- **Epochs:** 35,000
- **Learning rate:** 1e-3
- **Batch size:** 4,096
- **Collocation points:** 15,000 per Reynolds number (105,000 total)
- **Cd loss activation:** Starts at epoch 8,000 (after physics is learned first)
- **Checkpoint frequency:** Every 3,000 epochs
- **Print frequency:** Every 1,000 epochs
- **Wake/outer flow resampling:** Every 1,000 epochs

### Phase 2: L-BFGS Fine-Tuning (Round 1)
- **Iterations:** 3,000
- **Loss:** Physics + BC only (stable convergence)
- **β_BC:** 6.0
- **Method:** TensorFlow Probability L-BFGS or SciPy L-BFGS-B fallback

### Phase 3: Extended L-BFGS (Round 2)
- **Iterations:** 3,000 additional
- **Loss:** Physics + BC + **Cd** (adds drag coefficient accuracy)
- **β_Cd:** 1.0 (increased for final Cd accuracy)
- **Tolerance:** 1e-9 (tighter convergence)
- **Purpose:** Push Cd error below 3% across all Re values

```
Total Training: Adam (35k epochs) → L-BFGS R1 (3k iter) → L-BFGS R2 (3k iter)
```

---

## Repository Structure

```
parametric-pinn-flow-cylinder/
│
├── src/                        # Core source code
│   ├── model.py                # PINN architecture (8×40 network)
│   ├── train.py                # Full 3-phase training pipeline
│   ├── physics.py              # Navier-Stokes PDE residuals
│   ├── boundary.py             # Boundary condition enforcement
│   └── utils.py                # Visualization, domain sampling
│
├── notebooks/                  # Jupyter notebooks
│   ├── sem-results.ipynb       # Full training + results (Kaggle)
│   ├── 01_training.ipynb       # Model training walkthrough
│   ├── 02_evaluation.ipynb     # Validation and metrics
│   └── 03_visualization.ipynb  # Flow field plots
│
├── configs/                    # Training configurations
│   └── config.yaml             # Hyperparameters (Re range, β weights, lr)
│
├── results/                    # Output plots and saved models
│   ├── figures/                # Velocity/pressure field plots
│   └── checkpoints/            # Saved model weights (.weights.h5)
│
├── data/                       # Benchmark data (Dennis & Chang 1970)
│   └── README.md               # Reference data sources
│
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| TensorFlow 2.x | Neural network + automatic differentiation |
| TensorFlow Probability | L-BFGS optimizer (primary) |
| SciPy | L-BFGS-B optimizer (fallback) |
| NumPy | Numerical computation |
| Matplotlib / Seaborn | Flow field visualization |
| Kaggle (NVIDIA T4 GPU) | Training platform |

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
import tensorflow as tf

# Load trained model
model = ParametricPINN.load('results/checkpoints/best_model')

# Predict flow field at Re = 25
X = tf.constant([[0.5, 0.2, 25.0]], dtype=tf.float32)  # x, y, Re

# Get predictions
u, v, p, psi = model.compute_velocities_from_streamfunction(X)

print(f"Velocity: u={u.numpy()[0,0]:.4f}, v={v.numpy()[0,0]:.4f}")
print(f"Pressure: p={p.numpy()[0,0]:.4f}")
```

---

## Results & Validation

### Accuracy Comparison: Physics-Only vs Benchmark-Informed

| Training Mode | Training Re | Interpolation Re | Extrapolation Re |
|---------------|-------------|------------------|------------------|
| **Physics-Only (Mode A)** | 95–96% accuracy<br>(4–5% error) | 94–95% accuracy | 92–93% accuracy |
| **Benchmark-Informed (Mode B)** | **97–98% accuracy**<br>**(1–2% error)** | **97–98% accuracy** | **95–96% accuracy** |

### Drag Coefficient (Cd) Validation — Mode B
Compared against **Dennis & Chang (1970)** experimental data:

| Re | PINN Prediction | Benchmark | Error |
|----|----------------|-----------|-------|
| 10 | 2.94 | 2.96 | 0.7% |
| 20 | 2.03 | 2.05 | 1.0% |
| 30 | 1.87 | 1.89 | 1.1% |
| 40 | 1.57 | 1.59 | 1.3% |
| 28 (interp) | 1.66 | 1.72 | 3.3% |

### Key Insights
- **Physics-only baseline** achieves 95–96% accuracy with zero external data
- **Benchmark-informed constraints** improve accuracy to 97–98% by refining integral quantities (Cd, wake length)
- **Both approaches avoid CFD solution fields** — only physics equations and optional scalar targets

### Flow Field Visualization
*(Figures will be added — velocity fields, pressure contours, loss convergence curves)*

---

## Project Status

- [x] Baseline PINN for single Reynolds number
- [x] Parametric extension with Re as input (3D input space)
- [x] Physics-only training (Mode A: 95–96% accuracy)
- [x] Benchmark-informed training (Mode B: 97–98% accuracy)
- [x] 7 training + 2 interpolation + 2 extrapolation Re values
- [x] Multi-objective loss (6 terms: PDE + BC + Cd + Wake + Bernoulli + Surface)
- [x] 3-phase training: Adam → L-BFGS R1 → L-BFGS R2
- [ ] Add flow field visualization plots to `results/figures/`
- [ ] Extend to higher Re with time-dependent unsteady solver
- [ ] 3D extension (flow over sphere)
- [ ] Inverse problem: infer Re from velocity measurements

---

## Physics Regime: Steady Flow Only

This work focuses on **steady-state laminar flow** (Re < 47) to avoid unsteady vortex shedding. Key physics assumptions:

- **No time dependence** (∂/∂t = 0)
- **Steady Navier-Stokes equations** valid
- **Laminar regime** — no turbulence modeling needed
- **2D flow** — incompressible, constant properties

For Re > 47, the von Kármán vortex street emerges, requiring a time-dependent solver.

---

## Related Work

This project is part of ongoing M.Tech research at IIST exploring:
- **Inverse PINNs** for heat flux estimation in regenerative cooling channels
- **Parametric surrogate modeling** for aerodynamic design optimization
- **Physics-informed deep learning** for rocket propulsion thermal analysis

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{raghavendra2026parametricpinn,
  author    = {Raghavendra M},
  title     = {Parametric Physics-Informed Neural Networks for Steady Flow Over Cylinders},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/ragharit586-pixel/parametric-pinn-flow-cylinder},
  note      = {M.Tech Research, IIST}
}
```

---

## References

- **Dennis, S. C. R., & Chang, G. Z. (1970).** "Numerical solutions for steady flow past a circular cylinder at Reynolds numbers up to 100." *Journal of Fluid Mechanics*, 42(3), 471–489.
- **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686–707.
- **Rao, C., Sun, H., & Liu, Y. (2020).** "Physics-informed deep learning for incompressible laminar flows." *Theoretical and Applied Mechanics Letters*, 10(3), 207–212.

---

## Author

**Raghavendra M**  
M.Tech Aerospace Engineering (Thermal & Propulsion) @ IIST  
📧 ragharit586@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/raghavendra-mylar-b00b95240/)  
🐙 [GitHub](https://github.com/ragharit586-pixel)

---

<div align="center">
<sub>Built with Physics-Informed Neural Networks @ IIST | No CFD Solution Data Required</sub>
</div>
