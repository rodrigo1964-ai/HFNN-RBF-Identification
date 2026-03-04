"""
Generación de Figuras para Caso 3

Genera todas las figuras para el paper.

Author: CaseStudy_3 - Marzo 2026
"""
import numpy as np
import matplotlib.pyplot as plt
from rbf_integration import EntrenaRBFI, VectorRBFI
from caso3_regressor_rbf import (solve_ode_odeint, solve_ode_regressor_rbf,
                                  beta_true, compute_error)
from optimize_rbf_caso3 import objective_function, optimize_weights_nelder_mead


def figure_solution_comparison():
    """
    Figura 1: Comparación de soluciones
    """
    print("\n" + "="*70)
    print("Generando Figura 1: Comparación de soluciones")
    print("="*70)

    # Configuración
    y0 = -0.2
    n = 100
    t = np.linspace(-1, 1, n)

    # Resolver con odeint
    sol_odeint = solve_ode_odeint(y0, t)

    # Entrenar RBF
    p = 30
    x_train = np.linspace(-3, 2, p).reshape(-1, 1)
    y_train = beta_true(x_train)
    k = 5

    W, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)

    # Resolver con regresor
    sol_reg = solve_ode_regressor_rbf(sol_odeint[0], sol_odeint[1], t, W, centros, sigma)

    # Calcular errores
    errors = compute_error(sol_reg, sol_odeint)

    # Crear figura
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Panel superior: Soluciones
    ax1 = axes[0]
    ax1.plot(t, sol_odeint, 'b-', linewidth=2, label='odeint (referencia)')
    ax1.plot(t, sol_reg, 'ro', markersize=3, label='Regresor + RBF')
    ax1.set_xlabel('Tiempo t', fontsize=12)
    ax1.set_ylabel('y(t)', fontsize=12)
    ax1.set_title('Caso 3: y\' + β(y) = sin(5t), y(0) = -0.2', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Panel inferior: Error
    ax2 = axes[1]
    error_abs = np.abs(sol_reg - sol_odeint)
    ax2.plot(t, error_abs, 'r-', linewidth=1.5)
    ax2.set_xlabel('Tiempo t', fontsize=12)
    ax2.set_ylabel('Error absoluto |y_reg - y_ref|', fontsize=12)
    ax2.set_title(f'Error: Max={errors["max"]:.4e}, RMS={errors["rms"]:.4e}',
                  fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    filename = '/home/rodo/1Paper/CaseStudy_3/caso3_solution_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Guardada: {filename}")
    plt.close()


def figure_rbf_approximation():
    """
    Figura 2: Aproximación de β(y) con RBF
    """
    print("\n" + "="*70)
    print("Generando Figura 2: Aproximación de β(y)")
    print("="*70)

    # Datos de entrenamiento
    p = 30
    x_train = np.linspace(-3, 2, p).reshape(-1, 1)
    y_train = beta_true(x_train)

    # Datos de test
    x_test = np.linspace(-3.5, 2.5, 200).reshape(-1, 1)
    y_test = beta_true(x_test)

    # Entrenar con diferentes k
    k_values = [3, 5, 8]
    colors = ['orange', 'green', 'purple']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel izquierdo: Aproximaciones
    ax1 = axes[0]
    ax1.plot(x_test, y_test, 'b-', linewidth=2, label='β(y) real', zorder=10)
    ax1.plot(x_train, y_train, 'ko', markersize=5, label='Datos entrenamiento', zorder=11)

    for k, color in zip(k_values, colors):
        W, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)
        y_rbf = VectorRBFI(x_test, W, centros, sigma)
        rmse = np.sqrt(np.mean((y_rbf - y_test)**2))
        ax1.plot(x_test, y_rbf, color=color, linewidth=1.5,
                label=f'RBF k={k} (RMSE={rmse:.3e})', linestyle='--')

    ax1.set_xlabel('y', fontsize=12)
    ax1.set_ylabel('β(y)', fontsize=12)
    ax1.set_title('Aproximación de β(y) = 0.1y³ + 0.1y² + y - 1', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Panel derecho: Error vs k
    ax2 = axes[1]
    k_range = range(2, 16)
    rmse_values = []

    for k in k_range:
        W, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)
        y_rbf = VectorRBFI(x_test, W, centros, sigma)
        rmse = np.sqrt(np.mean((y_rbf - y_test)**2))
        rmse_values.append(rmse)

    ax2.plot(k_range, rmse_values, 'bo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Número de neuronas k', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('Error de aproximación vs k', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    filename = '/home/rodo/1Paper/CaseStudy_3/caso3_rbf_approximation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Guardada: {filename}")
    plt.close()


def figure_convergence_analysis():
    """
    Figura 3: Análisis de convergencia con N
    """
    print("\n" + "="*70)
    print("Generando Figura 3: Análisis de convergencia")
    print("="*70)

    y0 = -0.2
    n_values = [20, 30, 50, 75, 100, 150, 200, 300, 400]

    # Entrenar RBF
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

    print("\n  Calculando para diferentes N...")
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

        print(f"    N={n:3d}: Error RMS={errors['rms']:.4e}")

    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel izquierdo: Error vs N
    ax1 = axes[0]
    ax1.semilogy(results['n'], results['error_max'], 'ro-', linewidth=2,
                 markersize=6, label='Error máximo')
    ax1.semilogy(results['n'], results['error_rms'], 'bs-', linewidth=2,
                 markersize=6, label='Error RMS')
    ax1.set_xlabel('Número de puntos N', fontsize=12)
    ax1.set_ylabel('Error', fontsize=12)
    ax1.set_title('Convergencia vs Número de Puntos', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11)

    # Panel derecho: Error vs paso T
    ax2 = axes[1]
    ax2.loglog(results['T'], results['error_max'], 'ro-', linewidth=2,
               markersize=6, label='Error máximo')
    ax2.loglog(results['T'], results['error_rms'], 'bs-', linewidth=2,
               markersize=6, label='Error RMS')
    ax2.set_xlabel('Paso temporal T', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_title('Convergencia vs Paso Temporal', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=11)

    # Líneas de referencia (orden)
    T_ref = np.array(results['T'])
    ax2.plot(T_ref, T_ref**2 * results['error_max'][0] / results['T'][0]**2,
             'k--', alpha=0.5, linewidth=1, label='O(T²)')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    filename = '/home/rodo/1Paper/CaseStudy_3/caso3_convergence_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Guardada: {filename}")
    plt.close()


def figure_optimization_comparison():
    """
    Figura 4: Comparación optimización W_pinv vs W_opt
    """
    print("\n" + "="*70)
    print("Generando Figura 4: Comparación optimización")
    print("="*70)

    # Configuración
    y0 = -0.2
    n = 100
    t = np.linspace(-1, 1, n)

    # Resolver referencia
    sol_ref = solve_ode_odeint(y0, t)

    # Entrenar RBF
    p = 30
    x_train = np.linspace(-3, 2, p).reshape(-1, 1)
    y_train = beta_true(x_train)
    k = 5

    W_pinv, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)

    # Solución con W_pinv
    sol_pinv = solve_ode_regressor_rbf(sol_ref[0], sol_ref[1], t, W_pinv, centros, sigma)

    # Optimizar desde pesos ruidosos
    print("\n  Optimizando pesos...")
    noise = np.random.uniform(low=-1, high=1, size=W_pinv.shape)
    W_noisy = W_pinv + noise

    opt_result = optimize_weights_nelder_mead(W_noisy, sol_ref, k, t, centros, sigma, maxiter=200)
    W_opt = opt_result['W_opt']

    # Solución con W_opt
    sol_opt = solve_ode_regressor_rbf(sol_ref[0], sol_ref[1], t, W_opt, centros, sigma)

    # Calcular errores
    err_pinv = compute_error(sol_pinv, sol_ref)
    err_opt = compute_error(sol_opt, sol_ref)

    # Crear figura
    fig = plt.figure(figsize=(14, 10))

    # Panel 1: Soluciones completas
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, sol_ref, 'b-', linewidth=2, label='odeint (ref)')
    ax1.plot(t, sol_pinv, 'go', markersize=3, label=f'W_pinv (err={err_pinv["rms"]:.3e})')
    ax1.plot(t, sol_opt, 'r^', markersize=3, label=f'W_opt (err={err_opt["rms"]:.3e})')
    ax1.set_xlabel('Tiempo t', fontsize=11)
    ax1.set_ylabel('y(t)', fontsize=11)
    ax1.set_title('Comparación de Soluciones', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Panel 2: Errores
    ax2 = plt.subplot(2, 2, 2)
    err_abs_pinv = np.abs(sol_pinv - sol_ref)
    err_abs_opt = np.abs(sol_opt - sol_ref)
    ax2.semilogy(t, err_abs_pinv, 'g-', linewidth=1.5, label='W_pinv')
    ax2.semilogy(t, err_abs_opt, 'r-', linewidth=1.5, label='W_opt')
    ax2.set_xlabel('Tiempo t', fontsize=11)
    ax2.set_ylabel('Error absoluto', fontsize=11)
    ax2.set_title('Comparación de Errores', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)

    # Panel 3: Aproximación β(y)
    ax3 = plt.subplot(2, 2, 3)
    y_plot = np.linspace(-3, 2, 200).reshape(-1, 1)
    beta_real = beta_true(y_plot)
    beta_pinv = VectorRBFI(y_plot, W_pinv, centros, sigma)
    beta_opt = VectorRBFI(y_plot, W_opt, centros, sigma)

    ax3.plot(y_plot, beta_real, 'b-', linewidth=2, label='β(y) real')
    ax3.plot(y_plot, beta_pinv, 'g--', linewidth=1.5, label='β_RBF (W_pinv)')
    ax3.plot(y_plot, beta_opt, 'r--', linewidth=1.5, label='β_RBF (W_opt)')
    ax3.plot(x_train, y_train, 'ko', markersize=4, label='Datos')
    ax3.set_xlabel('y', fontsize=11)
    ax3.set_ylabel('β(y)', fontsize=11)
    ax3.set_title('Aproximación de β(y)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    # Panel 4: Comparación de pesos
    ax4 = plt.subplot(2, 2, 4)
    indices = np.arange(k)
    width = 0.35
    ax4.bar(indices - width/2, W_pinv.ravel(), width, label='W_pinv', color='green', alpha=0.7)
    ax4.bar(indices + width/2, W_opt.ravel(), width, label='W_opt', color='red', alpha=0.7)
    ax4.set_xlabel('Índice del peso', fontsize=11)
    ax4.set_ylabel('Valor del peso', fontsize=11)
    ax4.set_title('Comparación de Pesos', fontsize=12, fontweight='bold')
    ax4.set_xticks(indices)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=10)

    plt.tight_layout()
    filename = '/home/rodo/1Paper/CaseStudy_3/caso3_optimization_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Guardada: {filename}")
    plt.close()


def generate_all_figures():
    """
    Generar todas las figuras
    """
    print("\n" + "🎨"*35)
    print("GENERACIÓN DE FIGURAS - CASO 3")
    print("🎨"*35)

    figure_solution_comparison()
    figure_rbf_approximation()
    figure_convergence_analysis()
    figure_optimization_comparison()

    print("\n" + "="*70)
    print("✓ TODAS LAS FIGURAS GENERADAS")
    print("="*70)

    print("\n📁 Archivos generados:")
    print("   • caso3_solution_comparison.png")
    print("   • caso3_rbf_approximation.png")
    print("   • caso3_convergence_analysis.png")
    print("   • caso3_optimization_comparison.png")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    np.random.seed(42)
    generate_all_figures()
