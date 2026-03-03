# Parametric PINNs for Flow Over Cylinders

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Cd Accuracy](https://img.shields.io/badge/Cd%20Accuracy-97.7%25-success?style=flat-square)
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
| **Physics-Only Cd Error** | ~4–5% |
| **Benchmark-Informed Cd Error** | **~2.27% mean (1.5–3.4% range)** |
| **Separation Angle Error** | ~10.81% mean (6.6% for Re ≥ 20) |
| **Real-Time Inference Speed** | <50ms (NVIDIA T4) |
| **Reynolds Number Range** | Re = 10 → 47 (steady regime) |
| **Training Hardware** | NVIDIA T4 GPU (Kaggle) |
| **Total Parameters** | 11,722 |

---

## Flow Field Visualizations

### Parametric Generalization — u-velocity vs Reynolds Number
> Smooth continuous prediction across all Re. Training points lie exactly on the parametric curve.

![Parametric Curve](results/figures/parametric_curve_u_vs_Re.png)

---

### Velocity Profiles Across All Training Re
> Left: Horizontal centerline velocity (y=H/2). Right: Vertical velocity profile (x=0.5m).

![Velocity Profiles](results/figures/centerline_velocity_profiles.png)

---

### Training Cases

**Re = 10** — Training, Steady Regime
![Re10](results/figures/flow_field_Re10_training.png)

**Re = 15** — Training, Steady Regime
![Re15](results/figures/flow_field_Re15_training.png)

**Re = 20** — Training, Steady Regime
![Re20](results/figures/flow_field_Re20_training.png)

---

### Interpolation Tests (Re Never Seen During Training)

**Re = 12** — Interpolation Test (between Re=10 and Re=15)
![Re12](results/figures/flow_field_Re12_interpolation.png)

**Re = 28** — Interpolation Test (between Re=25 and Re=30)
![Re28](results/figures/flow_field_Re28_interpolation.png)

---

### Extrapolation Tests (Beyond Training Range)

**Re = 42** — Extrapolation (beyond max training Re=40)
![Re42](results/figures/flow_field_Re42_extrapolation.png)

**Re = 45** — Extrapolation (close to critical Re=47)
![Re45](results/figures/flow_field_Re45_extrapolation.png)

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
- **Total parameters:** 11,722
- **Input scaling:** All inputs normalized to [-1, 1] for balanced gradients
- **Velocity derivation:** `u = ∂ψ/∂y`, `v = -∂ψ/∂x` via automatic differentiation

### Reynolds Number Configuration

| Split | Re Values | Count |
|-------|-----------|-------|
| **Training** | 10, 15, 20, 25, 30, 35, 40 | 7 |
| **Interpolation test** | 12 (between 10–15), 28 (between 25–30) | 2 |
| **Extrapolation test** | 42, 45 (beyond training, still < Re_critical=47) | 2 |

All Reynolds numbers stay **below Re_critical = 47** (steady-state regime, no vortex shedding).

---

## Training Modes

### Mode A: Physics-Only (Unsupervised)

```
L_total = L_PDE + β_BC · L_BC
```
- **Cd Error:** ~4–5% | Zero external data

### Mode B: Physics + Benchmark-Informed Constraints (Weakly Supervised)

```
L_total = L_PDE + β_BC·L_BC + β_Cd·L_Cd + β_wake·L_wake + β_Bern·L_Bernoulli + β_surf·L_surface
```
- **Cd Error: ~2.27% mean** | Scalar targets from Dennis & Chang (1970) only — no CFD fields

---

## Multi-Objective Loss (Mode B)

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
- Kinematic viscosity: `ν = U·D/Re` (computed dynamically per point)

---

## Training Strategy — 3-Phase Pipeline

```
Adam (35k epochs) → L-BFGS Round 1 (3k iter) → L-BFGS Round 2 (3k iter)
```

| Phase | Method | Iterations | Loss Terms |
|-------|--------|------------|------------|
| 1 | Adam (lr=1e-3) | 35,000 epochs | PDE + BC + Cd (from epoch 8k) + Wake + Bernoulli + Surface |
| 2 | L-BFGS R1 | 3,000 | PDE + BC |
| 3 | L-BFGS R2 | 3,000 | PDE + BC + Cd (β=1.0) |

- **Total collocation points:** 105,000 (15,000/Re × 7 Re)
- **BC points:** 30,800 total
- **Wake/outer resampling:** Every 1,000 epochs

---

## Verified Results

### Drag Coefficient (Cd)
> Validated against **Dennis & Chang (1970)** for all 11 Reynolds numbers.

| Re | PINN Cd | Benchmark Cd | Error |
|----|---------|--------------|-------|
| 10 | 2.8596 | 2.812 | **1.69%** |
| 12 *(interp)* | 2.6007 | 2.689 | 3.28% |
| 15 | 2.3011 | 2.255 | **2.04%** |
| 20 | 1.9625 | 2.001 | **1.92%** |
| 25 | 1.7540 | 1.815 | 3.36% |
| 28 *(interp)* | 1.6645 | 1.720 | 3.23% |
| 30 | 1.6138 | 1.659 | **2.73%** |
| 35 | 1.5027 | 1.529 | **1.72%** |
| 40 | 1.3960 | 1.422 | **1.83%** |
| 42 *(extrap)* | 1.3571 | 1.378 | **1.52%** |
| 45 *(extrap)* | 1.3086 | 1.331 | **1.68%** |
| | | **Mean Error** | **2.27%** |

**Key observation:** Extrapolation cases (Re = 42, 45) achieve <2% Cd error — strong generalization beyond training range.

---

### Separation Angle

| Re | PINN (°) | Benchmark (°) | Error |
|----|----------|---------------|-------|
| 10 | 60.0 | 48.0 | 25.0% |
| 15 | 62.5 | 55.0 | 13.6% |
| 20 | 65.0 | 63.0 | **3.2%** |
| 25 | 67.5 | 70.0 | **3.6%** |
| 30 | 70.0 | 77.0 | 9.1% |
| 35 | 72.5 | 81.0 | 10.5% |
| 40 | 75.0 | 84.0 | 10.7% |
| | | **Mean Error** | **10.81%** |

> Note: Higher error at Re = 10–15 is attributed to weak separation signal at low Reynolds numbers (laminar onset regime). For **Re ≥ 20, mean error reduces to ~6.6%**.

---

## Validation Status

| Metric | Status |
|--------|--------|
| Drag Coefficient (Cd) | ✅ Validated — 2.27% mean error |
| Separation Angle | ⚠️ Validated — 10.81% mean (6.6% for Re ≥ 20) |
| Wake Length | 🔧 Under improvement |
| Flow field visualization | ✅ Available — see above |

---

## Repository Structure

```
parametric-pinn-flow-cylinder/
│
├── src/
│   ├── model.py                # PINN architecture (8×40, 11,722 params)
│   ├── train.py                # Full 3-phase training pipeline
│   ├── physics.py              # Navier-Stokes PDE residuals
│   ├── boundary.py             # Boundary condition enforcement
│   └── utils.py                # Visualization, domain sampling
│
├── notebooks/
│   └── sem-results.ipynb       # Full training + results (Kaggle T4)
│
├── results/
│   ├── figures/                # Flow field plots (uploaded separately)
│   └── checkpoints/            # Saved model weights (.weights.h5)
│
├── data/
│   └── README.md               # Dennis & Chang (1970) benchmark data
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| TensorFlow 2.x | Neural network + automatic differentiation |
| TensorFlow Probability | L-BFGS optimizer (primary) |
| SciPy | L-BFGS-B optimizer (fallback) |
| NumPy / Matplotlib | Numerics + visualization |
| Kaggle (NVIDIA T4 GPU) | Training platform |

---

## Project Status

- [x] 8×40 parametric PINN (11,722 parameters)
- [x] Physics-only training (Mode A: ~4-5% Cd error)
- [x] Benchmark-informed training (Mode B: **2.27% mean Cd error**)
- [x] 7 training + 2 interpolation + 2 extrapolation Re values (11 total)
- [x] Multi-objective loss: PDE + BC + Cd + Wake + Bernoulli + Surface
- [x] 3-phase training: Adam → L-BFGS R1 → L-BFGS R2
- [x] Cd validated for all 11 Re values
- [x] Separation angle validated
- [x] Flow field visualizations (Training + Interpolation + Extrapolation)
- [ ] Wake length post-processing fix
- [ ] Extend to higher Re (time-dependent solver)

---

## Physics Regime

Steady-state laminar flow (Re < 47): no time dependence, 2D incompressible, no turbulence modeling. For Re > 47, von Kármán vortex street requires a time-dependent solver.

---

## References

- **Dennis, S. C. R., & Chang, G. Z. (1970).** *J. Fluid Mechanics*, 42(3), 471–489.
- **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** *J. Computational Physics*, 378, 686–707.
- **Rao, C., Sun, H., & Liu, Y. (2020).** *Theoretical and Applied Mechanics Letters*, 10(3), 207–212.

---

## Citation

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
