"""
Figure Generation for Case Study 3

Generates all figures for the academic paper.

Author: CaseStudy_3 - March 2026
"""
import numpy as np
import matplotlib.pyplot as plt
from rbf_integration import EntrenaRBFI, VectorRBFI
from caso3_regressor_rbf import (solve_ode_odeint, solve_ode_regressor_rbf,
                                  beta_true, compute_error)
from optimize_rbf_caso3 import objective_function, optimize_weights_nelder_mead


def figure_solution_comparison():
    """
    Figure 1: Solution comparison between numerical reference and homotopic regressor with RBF
    """
    print("\n" + "="*70)
    print("Generating Figure 1: Solution Comparison")
    print("="*70)

    # Configuration
    y0 = -0.2
    n = 100
    t = np.linspace(-1, 1, n)

    # Solve with odeint (reference)
    sol_odeint = solve_ode_odeint(y0, t)

    # Train RBF network
    p = 30
    x_train = np.linspace(-3, 2, p).reshape(-1, 1)
    y_train = beta_true(x_train)
    k = 5

    W, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)

    # Solve with homotopic regressor
    sol_reg = solve_ode_regressor_rbf(sol_odeint[0], sol_odeint[1], t, W, centros, sigma)

    # Compute errors
    errors = compute_error(sol_reg, sol_odeint)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Upper panel: Solutions
    ax1 = axes[0]
    ax1.plot(t, sol_odeint, 'b-', linewidth=2, label='Numerical reference (odeint)')
    ax1.plot(t, sol_reg, 'ro', markersize=3, label='Homotopic regressor with RBF')
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('y(t)', fontsize=12)
    ax1.set_title('Case 3: $y\' + \\beta(y) = \\sin(5t)$ with RBF approximation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Lower panel: Error
    ax2 = axes[1]
    error_abs = np.abs(sol_reg - sol_odeint)
    ax2.plot(t, error_abs, 'r-', linewidth=1.5)
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel('Absolute error $|y_{reg} - y_{ref}|$', fontsize=12)
    ax2.set_title(f'Error metrics: Max = {errors["max"]:.4e}, RMS = {errors["rms"]:.4e}',
                  fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    filename = '/home/rodo/1Paper/CaseStudy_3/case3_solution_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


def figure_rbf_approximation():
    """
    Figure 2: RBF approximation of nonlinear function β(y)
    """
    print("\n" + "="*70)
    print("Generating Figure 2: RBF Approximation of β(y)")
    print("="*70)

    # Training data
    p = 30
    x_train = np.linspace(-3, 2, p).reshape(-1, 1)
    y_train = beta_true(x_train)

    # Test data
    x_test = np.linspace(-3.5, 2.5, 200).reshape(-1, 1)
    y_test = beta_true(x_test)

    # Train with different k values
    k_values = [3, 5, 8]
    colors = ['orange', 'green', 'purple']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Approximations
    ax1 = axes[0]
    ax1.plot(x_test, y_test, 'b-', linewidth=2, label='True function $\\beta(y)$', zorder=10)
    ax1.plot(x_train, y_train, 'ko', markersize=5, label='Training data', zorder=11)

    for k, color in zip(k_values, colors):
        W, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)
        y_rbf = VectorRBFI(x_test, W, centros, sigma)
        rmse = np.sqrt(np.mean((y_rbf - y_test)**2))
        ax1.plot(x_test, y_rbf, color=color, linewidth=1.5,
                label=f'RBF approximation (k={k}, RMSE={rmse:.3e})', linestyle='--')

    ax1.set_xlabel('y', fontsize=12)
    ax1.set_ylabel('$\\beta(y)$', fontsize=12)
    ax1.set_title('RBF Approximation: $\\beta(y) = 0.1y^3 + 0.1y^2 + y - 1$', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Right panel: Error vs k
    ax2 = axes[1]
    k_range = range(2, 16)
    rmse_values = []

    for k in k_range:
        W, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)
        y_rbf = VectorRBFI(x_test, W, centros, sigma)
        rmse = np.sqrt(np.mean((y_rbf - y_test)**2))
        rmse_values.append(rmse)

    ax2.plot(k_range, rmse_values, 'bo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of RBF neurons k', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('Approximation Error vs Number of Neurons', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    filename = '/home/rodo/1Paper/CaseStudy_3/case3_rbf_approximation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


def figure_convergence_analysis():
    """
    Figure 3: Convergence analysis with respect to number of discretization points
    """
    print("\n" + "="*70)
    print("Generating Figure 3: Convergence Analysis")
    print("="*70)

    y0 = -0.2
    n_values = [20, 30, 50, 75, 100, 150, 200, 300, 400]

    # Train RBF network
    p = 30
    x_train = np.linspace(-3, 2, p).reshape(-1, 1)
    y_train = beta_true(x_train)
    k = 5
    W, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)

    results = {
        'n': [],
        'T': [],
        'error_max': [],
        'error_rms': []
    }

    print("\n  Computing for different N values...")
    for i, n in enumerate(n_values):
        t = np.linspace(-1, 1, n)
        T = t[1] - t[0]

        sol_ref = solve_ode_odeint(y0, t)
        sol_reg = solve_ode_regressor_rbf(sol_ref[0], sol_ref[1], t, W, centros, sigma)

        errors = compute_error(sol_reg, sol_ref)

        results['n'].append(n)
        results['T'].append(T)
        results['error_max'].append(errors['max'])
        results['error_rms'].append(errors['rms'])

        print(f"    N={n:3d}: RMS Error = {errors['rms']:.4e}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Error vs N
    ax1 = axes[0]
    ax1.semilogy(results['n'], results['error_max'], 'ro-', linewidth=2,
                 markersize=6, label='Maximum error')
    ax1.semilogy(results['n'], results['error_rms'], 'bs-', linewidth=2,
                 markersize=6, label='RMS error')
    ax1.set_xlabel('Number of discretization points N', fontsize=12)
    ax1.set_ylabel('Error', fontsize=12)
    ax1.set_title('Convergence vs Number of Points', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11)

    # Right panel: Error vs time step T
    ax2 = axes[1]
    ax2.loglog(results['T'], results['error_max'], 'ro-', linewidth=2,
               markersize=6, label='Maximum error')
    ax2.loglog(results['T'], results['error_rms'], 'bs-', linewidth=2,
               markersize=6, label='RMS error')
    ax2.set_xlabel('Time step T', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_title('Convergence vs Time Step', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=11)

    # Reference lines (convergence order)
    T_ref = np.array(results['T'])
    ax2.plot(T_ref, T_ref**2 * results['error_max'][0] / results['T'][0]**2,
             'k--', alpha=0.5, linewidth=1, label='$O(T^2)$')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    filename = '/home/rodo/1Paper/CaseStudy_3/case3_convergence_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


def figure_optimization_comparison():
    """
    Figure 4: Comparison between pseudo-inverse and optimized RBF weights
    """
    print("\n" + "="*70)
    print("Generating Figure 4: Optimization Comparison")
    print("="*70)

    # Configuration
    y0 = -0.2
    n = 100
    t = np.linspace(-1, 1, n)

    # Solve reference
    sol_ref = solve_ode_odeint(y0, t)

    # Train RBF network
    p = 30
    x_train = np.linspace(-3, 2, p).reshape(-1, 1)
    y_train = beta_true(x_train)
    k = 5

    W_pinv, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)

    # Solution with pseudo-inverse weights
    sol_pinv = solve_ode_regressor_rbf(sol_ref[0], sol_ref[1], t, W_pinv, centros, sigma)

    # Optimize from noisy initial weights
    print("\n  Optimizing RBF weights...")
    noise = np.random.uniform(low=-1, high=1, size=W_pinv.shape)
    W_noisy = W_pinv + noise

    opt_result = optimize_weights_nelder_mead(W_noisy, sol_ref, k, t, centros, sigma, maxiter=200)
    W_opt = opt_result['W_opt']

    # Solution with optimized weights
    sol_opt = solve_ode_regressor_rbf(sol_ref[0], sol_ref[1], t, W_opt, centros, sigma)

    # Compute errors
    err_pinv = compute_error(sol_pinv, sol_ref)
    err_opt = compute_error(sol_opt, sol_ref)

    # Create figure
    fig = plt.figure(figsize=(14, 10))

    # Panel 1: Complete solutions
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, sol_ref, 'b-', linewidth=2, label='Numerical reference')
    ax1.plot(t, sol_pinv, 'go', markersize=3, label=f'Pseudo-inverse (RMS={err_pinv["rms"]:.3e})')
    ax1.plot(t, sol_opt, 'r^', markersize=3, label=f'Optimized weights (RMS={err_opt["rms"]:.3e})')
    ax1.set_xlabel('Time t', fontsize=11)
    ax1.set_ylabel('y(t)', fontsize=11)
    ax1.set_title('Solution Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Panel 2: Errors
    ax2 = plt.subplot(2, 2, 2)
    err_abs_pinv = np.abs(sol_pinv - sol_ref)
    err_abs_opt = np.abs(sol_opt - sol_ref)
    ax2.semilogy(t, err_abs_pinv, 'g-', linewidth=1.5, label='Pseudo-inverse')
    ax2.semilogy(t, err_abs_opt, 'r-', linewidth=1.5, label='Optimized')
    ax2.set_xlabel('Time t', fontsize=11)
    ax2.set_ylabel('Absolute error', fontsize=11)
    ax2.set_title('Error Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)

    # Panel 3: β(y) approximation
    ax3 = plt.subplot(2, 2, 3)
    y_plot = np.linspace(-3, 2, 200).reshape(-1, 1)
    beta_real = beta_true(y_plot)
    beta_pinv = VectorRBFI(y_plot, W_pinv, centros, sigma)
    beta_opt = VectorRBFI(y_plot, W_opt, centros, sigma)

    ax3.plot(y_plot, beta_real, 'b-', linewidth=2, label='True function $\\beta(y)$')
    ax3.plot(y_plot, beta_pinv, 'g--', linewidth=1.5, label='RBF (pseudo-inverse)')
    ax3.plot(y_plot, beta_opt, 'r--', linewidth=1.5, label='RBF (optimized)')
    ax3.plot(x_train, y_train, 'ko', markersize=4, label='Training data')
    ax3.set_xlabel('y', fontsize=11)
    ax3.set_ylabel('$\\beta(y)$', fontsize=11)
    ax3.set_title('Function Approximation Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    # Panel 4: Weight comparison
    ax4 = plt.subplot(2, 2, 4)
    indices = np.arange(k)
    width = 0.35
    ax4.bar(indices - width/2, W_pinv.ravel(), width, label='Pseudo-inverse', color='green', alpha=0.7)
    ax4.bar(indices + width/2, W_opt.ravel(), width, label='Optimized', color='red', alpha=0.7)
    ax4.set_xlabel('Weight index', fontsize=11)
    ax4.set_ylabel('Weight value', fontsize=11)
    ax4.set_title('RBF Weight Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(indices)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=10)

    plt.tight_layout()
    filename = '/home/rodo/1Paper/CaseStudy_3/case3_optimization_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


def generate_all_figures():
    """
    Generate all figures for the academic paper
    """
    print("\n" + "="*70)
    print("FIGURE GENERATION - CASE STUDY 3")
    print("="*70)

    figure_solution_comparison()
    figure_rbf_approximation()
    figure_convergence_analysis()
    figure_optimization_comparison()

    print("\n" + "="*70)
    print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*70)

    print("\n📁 Generated files:")
    print("   • case3_solution_comparison.png")
    print("   • case3_rbf_approximation.png")
    print("   • case3_convergence_analysis.png")
    print("   • case3_optimization_comparison.png")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    np.random.seed(42)
    generate_all_figures()
