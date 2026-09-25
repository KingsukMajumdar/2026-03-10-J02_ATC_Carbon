<div align="center">

<h1>⚡ Reliability-Guided ATC Assessment using DNN Surrogate</h1>

<p><strong>IEEE Transactions on Reliability -- Special Section TREL-RRMP</strong> &nbsp;|&nbsp; Manuscript No. TR-2026-712 (Under review) </p>

<a href="https://doi.org/10.5281/zenodo.21607047"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.21607047.svg" alt="DOI"></a>&nbsp;
<a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg" alt="License"></a>&nbsp;
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.13-3776AB.svg?logo=python&logoColor=white" alt="Python"></a>&nbsp;
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>&nbsp;
<img src="https://img.shields.io/badge/Platform-Linux%20-informational.svg" alt="Platform">

</div>

---

## 📋 About

This repository provides **data, pre-trained DNN surrogate model weights,
simulation results, and validation figures** for the paper:

> **"Reliability-Guided Available Transfer Capability Assessment
> using Deep Neural Network Surrogate: Stochastic Renewable Energy
> Uncertainty and Carbon Emission Penalty"**
>
> K. Majumdar · S. Ghosh · N. Kumar
> *IEEE Transactions on Reliability (TREL-RRMP)*, 2026

> **Version 3.0** -- Death penalty revision (September 2026).
> Previous version 2.0 used a soft VaR penalty; v3.0 implements a
> hard death penalty ($F = -\infty$ for infeasible solutions).

### ✨ Key Highlights

| Feature | Detail |
|---------|--------|
| 🤖 DNN Surrogate | R² = 0.9975 / 0.9675 / 0.9999 on three power systems |
| ⚡ Speedup | ~1400× faster than full AC power flow solver |
| 🔒 Hard Constraint | Death penalty: VaR₉₀ < ATC_min → F = −∞ (rejected) |
| 🌿 Carbon-Aware | Soft guidance penalty term in fitness function |
| 📊 Reliability | Hard VaR₉₀ constraint: ATC_min = 0.62 / 0.25 / 0.30 PU |
| 🗺️ Systems | IEEE 30-bus · IEEE 118-bus · NRPG 246-bus (India) |
| 🔍 Validation | 500 independent out-of-sample NREL scenarios |

---

## 🗂️ Repository Structure

```
📦 TR-2026-712
 ┣ 📂 FIGURES/                         All validation figures
 ┃  ┣ S07_fig01/02/04                  DNN error histograms + scatter
 ┃  ┣ S08_fig01/02/03                  Out-of-sample validation charts
 ┃  ┣ S09_fig01/02/03                  Carbon sensitivity curves
 ┃  ┣ S10_fig01/02/03                  N-1 contingency analysis
 ┃  ┣ S11_fig01/02/03                  Surrogate model comparison
 ┃  └ fig1/2/3/4                       Manuscript figures (v3.0 ATC_min)
 ┣ 🧠 surrogate_case30.pt              Pre-trained DNN — IEEE 30-bus
 ┣ 🧠 surrogate_case118.pt             Pre-trained DNN — IEEE 118-bus
 ┣ 🧠 surrogate_nrpg246.pt             Pre-trained DNN — NRPG 246-bus
 ┣ 🗺️  NRPG246_full.json               NRPG 246-bus pandapower network
 ┣ 📊 S07_data02_dnn_validation_summary.csv
 ┣ 📊 S08_data02_oos_summary.csv
 ┣ 📊 S09_data02_carbon_sensitivity.csv
 ┣ 📊 S10_data02_n1_summary.csv
 ┣ 📊 S11_data02_model_comparison.csv
 ┣ 📄 S4_full_results.json             v3.0 results (death penalty)
 ┣ 📄 S6_nrpg246_results.json          NRPG 246-bus results
 └ 📖 README.md
```

---

## 🧠 Pre-trained DNN Surrogate Models

Three surrogate models are provided as PyTorch `.pt` files.
Load them **directly without retraining** to reproduce all DNN predictions.

| Model | System | Input dim | Architecture | R² | MAE (PU) |
|-------|--------|-----------|-------------|-----|---------|
| `surrogate_case30.pt` | IEEE 30-bus | 80 | [256, 512, 256, 128] | **0.9975** | 0.0037 |
| `surrogate_case118.pt` | IEEE 118-bus | 324 | [256, 512, 256, 128] | **0.9675** | 0.0239 |
| `surrogate_nrpg246.pt` | NRPG 246-bus | 554 | [512, 1024, 512, 256] | **0.9999** | 0.000270 |

### Quick Load Example

```python
import torch

ck = torch.load('surrogate_case30.pt',
                map_location='cpu',
                weights_only=False)

print(f"n_in   = {ck['n_in']}")       # 80
print(f"y_mean = {ck['y_mean']:.4f}") # 0.6484
print(f"y_std  = {ck['y_std']:.4f}")  # 0.1003
# Keys: state_dict, n_in, y_mean, y_std
```

---

## 📊 Simulation Results

### v3.0 -- Death Penalty Experiment (30 trials, moderate CE budget)

ATC_min = 0.62 PU (IEEE 30-bus), 0.25 PU (IEEE 118-bus), 0.30 PU (NRPG).
Infeasible solutions (VaR₉₀ < ATC_min) assigned F = −∞ and rejected.

**IEEE 30-bus -- all five algorithms: 100% feasibility**

| Algorithm | E[ATC] (PU) | VaR₉₀ (PU) | CE (kg/h) | Feasibility |
|-----------|------------|-----------|---------|------------|
| BBO | 0.7643 ± 0.0114 | 0.7045 | 275.3 | **100%** |
| GWO | **0.7752 ± 0.0059** | 0.7025 | **235.8** | **100%** |
| SHO | 0.6388 ± 0.0056 | 0.6364 | 471.2 | **100%** |
| CSHO | 0.6382 ± 0.0051 | 0.6357 | 464.8 | **100%** |
| OSHO | 0.6398 ± 0.0085 | 0.6375 | 452.0 | **100%** |

**IEEE 118-bus -- BBO and GWO achieve 100% feasibility**

| Algorithm | E[ATC] (PU) | VaR₉₀ (PU) | Feasibility |
|-----------|------------|-----------|------------|
| BBO | **0.7808 ± 0.0112** | 0.6460 | **100%** |
| GWO | 0.7232 ± 0.0251 | 0.5985 | **100%** |
| SHO | 0.2938 ± 0.0966 | 0.2680 | 60% |
| CSHO | 0.2788 ± 0.0894 | 0.2604 | 57% |
| OSHO | 0.3488 ± 0.0768 | 0.3136 | 90% |

*Feasibility rate = % of 30 trials in which a feasible solution was found.
E[ATC] and VaR₉₀ are averaged over feasible trials only.*

### Validation Results (unchanged from v2.0)

| File | Manuscript Table | Key Finding |
|------|-----------------|-------------|
| `S07_data02_dnn_validation_summary.csv` | Tab. I | R²=0.9975/0.9675/0.9999 |
| `S08_data02_oos_summary.csv` | Tab. VIII | Max OOS bias = +2.06% |
| `S09_data02_carbon_sensitivity.csv` | Tab. IX | Binding at 120% budget |
| `S10_data02_n1_summary.csv` | — | N-1 MAE ratio = 1.2× |
| `S11_data02_model_comparison.csv` | Tab. II | DNN beats RF/LR by 41-48% |

---

## 🔬 Results at a Glance

### DNN vs Baseline Models — IEEE 118-bus

| Model | R² | RMSE (PU) | MAE (PU) |
|-------|-----|-----------|---------|
| Linear Regression | 0.9096 | 0.0611 | 0.0406 |
| Random Forest | 0.9005 | 0.0641 | 0.0458 |
| **DNN (ours)** | **0.9675** | **0.0367** | **0.0239** |

### Out-of-Sample Validation — 500 Independent NREL Scenarios

| System | Max OOS Bias | VaR₉₀ Generalises |
|--------|-------------|------------------|
| IEEE 30-bus | +1.40% | ✅ |
| IEEE 118-bus | +2.06% | ✅ |
| NRPG 246-bus | +0.004% | ✅ |

---

## 🗺️ NRPG 246-Bus Network

`NRPG246_full.json` -- pandapower-compatible network for the
Northern Regional Power Grid of India:

- 246 buses · 612 branches · 42 generators · 19,994.8 MW load
- Monitored corridor: **Chamera--Jalandhar 400 kV** (bus 3 → bus 73)

> ⚠️ **Data attribution:** The original NRPG 246-bus data was sourced from
> the Power Systems Laboratory, IIT Kanpur, where it was publicly available
> during the period of this research. This file is a pandapower derivative
> for academic use only. Please acknowledge IIT Kanpur as the original source.

---

## ⚙️ Software Environment

| Package | Version |
|---------|---------|
| Python | 3.13.2 |
| PyTorch | 2.9.0 (CUDA 12.8) |
| pandapower | 3.x |
| NumPy | 2.3.3 |
| scikit-learn | latest |

**Hardware:** AMD Ryzen 9-8945HS · NVIDIA RTX 4060 (8 GB) · Manjaro Linux KDE

---

## 📜 Citation

```bibtex
@article{majumdar2026atc,
  author  = {Majumdar, Kingsuk and Ghosh, Sohini and Kumar, Nishant},
  title   = {Reliability-Guided Available Transfer Capability Assessment
             using Deep Neural Network Surrogate: Stochastic Renewable
             Energy Uncertainty and Carbon Emission Penalty},
  journal = {IEEE Transactions on Reliability},
  year    = {2026},
  doi     = {10.5281/zenodo.19753213}
}
```

---

## 🔗 Links

| | |
|-|-|
| 📦 Data (Zenodo v3.0) | https://doi.org/10.5281/zenodo.21607047 |
| 🏫 BCREC Durgapur | https://www.bcrec.ac.in |
| 🔬 ORCID | https://orcid.org/0000-0001-7224-4862 |
| 📧 Contact | kingsuk.majumdar@bcrec.ac.in |

---

## 📄 Licence

Results, model weights and figures → [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

`NRPG246_full.json` → Academic use only · Acknowledge IIT Kanpur

---

<div align="center">
<sub>🔬 Tesla Research Lab · Department of Electrical Engineering · BCREC Durgapur · 2026</sub>
</div>
