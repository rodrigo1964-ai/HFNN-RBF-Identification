# HFNN-RBF-Identification

**Homotopy-Based Functional Neural Networks with Radial Basis Functions for Robust Identification of Nonlinear Dynamic Systems**

> Companion code for the manuscript submitted to *Neurocomputing* (Elsevier), Manuscript ID: NEUCOM-D-25-16620.

## Overview

This repository contains the complete source code and reproducible experiments for a novel approach to nonlinear system identification using:

- **Homotopy Analysis Method (HAM)** for converting implicit nonlinear discrete equations into explicit algebraic regressors
- **Radial Basis Function (RBF) networks** as embedded approximators with analytical derivatives
- **Levenberg–Marquardt optimization** for parameter adaptation with as few as 30 data samples

The HFNN regressor uses $N(y)$, $N'(y)$, and $N''(y)$ — computed exactly from the Gaussian RBF kernel — to resolve the implicit nonlinear coupling algebraically, providing a theoretically founded discretization analogous to what Tustin and zero-order hold provide for linear systems.

## Repository Structure

```
├── PublicationA/    # RBF identification in first-order ODEs
│   ├── ode_rbf_identification.py
│   ├── ode_rbf_data_requirements.py
│   └── rbf_example.py
│
├── PublicationB/    # Direct optimization (no numerical derivatives)
│   ├── ode_rbf_direct_optimization.py
│   └── ode_rbf_direct_sensitivity.py
│
├── PublicationC/    # Pendulum with viscous friction (2nd order)
│   ├── pendulum_rbf_identification.py
│   └── pendulum_sensitivity_analysis.py
│
├── PublicationD/    # Duffing oscillator — limits of direct optimization
│   └── duffing_rbf_identification.py
│
├── PublicationE/    # Homotopy regressor with RBF for Duffing (main result)
│   ├── rbf_analytical.py              # RBF with analytical derivatives
│   ├── duffing_regressor_rbf.py       # Homotopy regressor implementation
│   ├── test_duffing_regressor.py      # Main experiments
│   ├── sensitivity_analysis_regressor.py
│   └── optimize_rbf_regressor.py      # LM optimization
│
└── *.py, *.png     # Additional scripts and figures
```

## Key Results

| Method | Samples | MSE | Time |
|--------|---------|-----|------|
| Traditional (numerical derivatives + RBF) | 40 | 2.7 × 10⁻¹ | 0.0004 s |
| HFNN regressor with embedded RBF | 30 | < 10⁻⁸ | 0.31 s |
| PINNs (literature) | 100–500 | ~ 10⁻³ | minutes |
| Neural ODEs (literature) | 300–1000 | ~ 10⁻³ | minutes |

## Requirements

```
Python >= 3.8
numpy
scipy
matplotlib
```

## Quick Start

```bash
# Run the main Duffing experiment with homotopy regressor
cd PublicationE
python test_duffing_regressor.py

# Run sensitivity analysis (varying number of data points)
python sensitivity_analysis_regressor.py

# Run RBF optimization with Levenberg-Marquardt
python optimize_rbf_regressor.py
```

## RBF Analytical Derivatives

The Gaussian RBF and its exact derivatives used in the homotopy regressor:

$$N(y) = \sum_{j=1}^{M} w_j \exp\left(-\frac{(y - c_j)^2}{2\sigma^2}\right) + w_0$$

$$N'(y) = \sum_{j=1}^{M} w_j \varphi_j(y) \cdot \left(-\frac{y - c_j}{\sigma^2}\right)$$

$$N''(y) = \sum_{j=1}^{M} w_j \varphi_j(y) \cdot \left(\frac{(y - c_j)^2}{\sigma^4} - \frac{1}{\sigma^2}\right)$$

These feed into the HFNN regressor: $y_k = y_{k-1} + z_1 + z_2$, where $z_1$ is the Newton correction and $z_2$ is the Halley/Olver correction.

## Citation

If you use this code, please cite:

```bibtex
@article{rodrigo2025hfnn,
  title={Homotopy-Based Functional Neural Networks for Robust Identification 
         of Uncertain Nonlinear Dynamic Systems},
  author={Rodrigo, Rodolfo H. and Schweickardt, Gustavo and Pati{\~n}o, Daniel H.},
  journal={Neurocomputing (submitted)},
  year={2025}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Authors

- **Rodolfo H. Rodrigo** — Universidad Nacional de San Juan
- **Gustavo Schweickardt** — CONICET / Universidad Tecnológica Nacional
- **Daniel H. Patiño** — Instituto de Automática (INAUT), UNSJ
