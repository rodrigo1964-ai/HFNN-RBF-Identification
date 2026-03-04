"""
generate_figures.py - Generate all figures for Case Study 2

Generates publication-quality figures for academic paper:
1. Comparison between homotopic regressor and odeint reference
2. Convergence analysis
3. Phase portrait (optional)

Author: Rodolfo H. Rodrigo - UNSJ
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from caso2_regressor import (
    solve_ode_regressor, solve_ode_rk4, compute_error
)

# Publication-style configuration
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100


def figure_1_comparison():
    """
    Figure 1: Direct comparison between homotopic regressor and odeint.

    Generates a two-panel figure showing:
    - Top panel: Solution trajectories
    - Bottom panel: Absolute error in logarithmic scale
    """
    print("Generating Figure 1: Comparison...")

    y0 = -0.2
    t_span = (0, 10)
    n = 500

    # Compute solutions
    t_rk4, y_rk4 = solve_ode_rk4(y0, t_span, n)
    t_reg, y_reg = solve_ode_regressor(y0, t_span, n)

    # Compute error metrics
    errors = compute_error(y_rk4, y_reg)
    error_abs = np.abs(y_rk4 - y_reg)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Subplot 1: Solution trajectories
    axes[0].plot(t_rk4, y_rk4, 'b-', linewidth=2, label='odeint (reference)')
    axes[0].plot(t_reg, y_reg, 'ro', markersize=2.5, label='Homotopic Regressor', alpha=0.7)
    axes[0].set_xlabel('Time $t$')
    axes[0].set_ylabel('$y(t)$')
    axes[0].set_title(r"Case 2: $y' + \sin^2(y) = \sin(5t)$ with $y(0) = -0.2$", fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.98, f'$N = {n}$ points', transform=axes[0].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Subplot 2: Absolute error
    axes[1].semilogy(t_rk4, error_abs, 'r-', linewidth=1.5)
    axes[1].set_xlabel('Time $t$')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title(f"Error: max = ${errors['max_error']:.2e}$, "
                     f"RMS = ${errors['rms_error']:.2e}$")
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].axhline(y=errors['rms_error'], color='orange', linestyle='--',
                   linewidth=1, label=f"RMS = {errors['rms_error']:.2e}")
    axes[1].legend(loc='best')

    plt.tight_layout()
    plt.savefig('case2_comparison_N500.png', dpi=300, bbox_inches='tight')
    print("  ✓ case2_comparison_N500.png")
    plt.close()


def figure_2_convergence():
    """
    Figure 2: Convergence analysis (Error vs. time step).

    Analyzes the convergence order by plotting error metrics
    against time step size in log-log scale.
    """
    print("Generating Figure 2: Convergence...")

    y0 = -0.2
    t_span = (0, 10)
    n_values = [100, 200, 500, 1000, 2000, 5000, 10000]

    T_values = []
    rms_errors = []
    max_errors = []

    for n in n_values:
        t_rk4, y_rk4 = solve_ode_rk4(y0, t_span, n)
        t_reg, y_reg = solve_ode_regressor(y0, t_span, n)
        errors = compute_error(y_rk4, y_reg)

        T = t_rk4[1] - t_rk4[0]
        T_values.append(T)
        rms_errors.append(errors['rms_error'])
        max_errors.append(errors['max_error'])

    # Convert to arrays
    T_values = np.array(T_values)
    rms_errors = np.array(rms_errors)
    max_errors = np.array(max_errors)

    # Estimate convergence order (log-log fit)
    from scipy.stats import linregress
    log_T = np.log(T_values)
    log_rms = np.log(rms_errors)
    slope, intercept, _, _, _ = linregress(log_T, log_rms)
    order = slope

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.loglog(T_values, rms_errors, 'bo-', linewidth=2, markersize=8,
              label='RMS Error', zorder=3)
    ax.loglog(T_values, max_errors, 'rs--', linewidth=2, markersize=8,
              label='Maximum Error', zorder=3)

    # Reference line with estimated slope
    T_fit = np.array([T_values[0], T_values[-1]])
    error_fit = np.exp(intercept) * T_fit**order
    ax.loglog(T_fit, error_fit, 'k:', linewidth=2,
              label=f'$O(T^{{{order:.2f}}})$', zorder=2)

    ax.set_xlabel('Time Step $T$')
    ax.set_ylabel('Error')
    ax.set_title("Case 2: Convergence of Homotopic Regressor", fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)

    # Add annotations
    ax.text(0.05, 0.05, f'Convergence order: $p \\approx {order:.2f}$',
           transform=ax.transAxes, fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig('case2_convergence.png', dpi=300, bbox_inches='tight')
    print("  ✓ case2_convergence.png")
    plt.close()


def figure_3_multiple_n():
    """
    Figure 3: Comparison for different values of N in subplots.

    Shows the effect of grid resolution on solution accuracy
    using three different values of N.
    """
    print("Generating Figure 3: Multiple N comparison...")

    y0 = -0.2
    t_span = (0, 10)
    n_values = [100, 500, 2000]

    fig, axes = plt.subplots(len(n_values), 1, figsize=(10, 10))

    for idx, n in enumerate(n_values):
        t_rk4, y_rk4 = solve_ode_rk4(y0, t_span, n)
        t_reg, y_reg = solve_ode_regressor(y0, t_span, n)
        errors = compute_error(y_rk4, y_reg)

        axes[idx].plot(t_rk4, y_rk4, 'b-', linewidth=2, label='odeint')
        axes[idx].plot(t_reg, y_reg, 'ro', markersize=2, label='Regressor', alpha=0.6)
        axes[idx].set_xlabel('Time $t$')
        axes[idx].set_ylabel('$y(t)$')
        axes[idx].set_title(f"$N = {n}$ | RMS Error = ${errors['rms_error']:.2e}$")
        axes[idx].legend(loc='best')
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle("Case 2: Effect of Grid Resolution $N$", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('case2_multiple_N.png', dpi=300, bbox_inches='tight')
    print("  ✓ case2_multiple_N.png")
    plt.close()


def figure_4_phase_portrait():
    """
    Figure 4: Phase portrait y(t) vs. y'(t).

    Visualizes the system dynamics in phase space,
    comparing the homotopic regressor with the reference solution.
    """
    print("Generating Figure 4: Phase portrait...")

    y0 = -0.2
    t_span = (0, 10)
    n = 2000

    # Compute solutions
    t_rk4, y_rk4 = solve_ode_rk4(y0, t_span, n)
    t_reg, y_reg = solve_ode_regressor(y0, t_span, n)

    # Compute derivatives (finite difference approximation)
    dydt_rk4 = np.gradient(y_rk4, t_rk4)
    dydt_reg = np.gradient(y_reg, t_reg)

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(y_rk4, dydt_rk4, 'b-', linewidth=2, label='odeint', alpha=0.7)
    ax.plot(y_reg, dydt_reg, 'r--', linewidth=1.5, label='Regressor', alpha=0.8)
    ax.plot(y0, dydt_rk4[0], 'go', markersize=10, label='Initial condition', zorder=5)

    ax.set_xlabel('$y$')
    ax.set_ylabel("$y'$")
    ax.set_title("Case 2: Phase Portrait", fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('case2_phase_portrait.png', dpi=300, bbox_inches='tight')
    print("  ✓ case2_phase_portrait.png")
    plt.close()


# ============================================================
# Generate all figures
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Generating all figures for Case Study 2")
    print("=" * 70)
    print()

    figure_1_comparison()
    figure_2_convergence()
    figure_3_multiple_n()
    figure_4_phase_portrait()

    print()
    print("=" * 70)
    print("All figures generated successfully ✓")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - case2_comparison_N500.png")
    print("  - case2_convergence.png")
    print("  - case2_multiple_N.png")
    print("  - case2_phase_portrait.png")
