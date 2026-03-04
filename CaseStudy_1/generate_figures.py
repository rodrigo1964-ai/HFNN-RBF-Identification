"""
generate_figures.py - Genera todas las figuras para el Caso 1

Genera figuras de alta calidad para publicación:
1. Comparación regresor vs odeint
2. Análisis de convergencia
3. Fase portrait (opcional)

Author: Rodolfo H. Rodrigo - UNSJ
Fecha: Marzo 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from caso1_regressor import (
    solve_ode_regressor, solve_ode_rk4, compute_error
)

# Configuración de estilo para publicación
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100


def figure_1_comparison():
    """
    Figura 1: Comparación directa regresor vs odeint.
    """
    print("Generando Figura 1: Comparación...")

    y0 = -0.2
    t_span = (0, 10)
    n = 500

    # Soluciones
    t_rk4, y_rk4 = solve_ode_rk4(y0, t_span, n)
    t_reg, y_reg = solve_ode_regressor(y0, t_span, n)

    # Calcular error
    errors = compute_error(y_rk4, y_reg)
    error_abs = np.abs(y_rk4 - y_reg)

    # Crear figura
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Subplot 1: Soluciones
    axes[0].plot(t_rk4, y_rk4, 'b-', linewidth=2, label='odeint (referencia)')
    axes[0].plot(t_reg, y_reg, 'ro', markersize=2.5, label='Regresor Homotópico', alpha=0.7)
    axes[0].set_xlabel('Tiempo $t$')
    axes[0].set_ylabel('$y(t)$')
    axes[0].set_title("Caso 1: $y' + y^2 = \sin(5t)$ con $y(0) = -0.2$", fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.98, f'$N = {n}$ puntos', transform=axes[0].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Subplot 2: Error absoluto
    axes[1].semilogy(t_rk4, error_abs, 'r-', linewidth=1.5)
    axes[1].set_xlabel('Tiempo $t$')
    axes[1].set_ylabel('Error absoluto')
    axes[1].set_title(f"Error: máx = ${errors['max_error']:.2e}$, "
                     f"RMS = ${errors['rms_error']:.2e}$")
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].axhline(y=errors['rms_error'], color='orange', linestyle='--',
                   linewidth=1, label=f"RMS = {errors['rms_error']:.2e}")
    axes[1].legend(loc='best')

    plt.tight_layout()
    plt.savefig('caso1_comparison_N500.png', dpi=300, bbox_inches='tight')
    print("  ✓ caso1_comparison_N500.png")
    plt.close()


def figure_2_convergence():
    """
    Figura 2: Análisis de convergencia (Error vs paso temporal).
    """
    print("Generando Figura 2: Convergencia...")

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

    # Convertir a arrays
    T_values = np.array(T_values)
    rms_errors = np.array(rms_errors)
    max_errors = np.array(max_errors)

    # Estimar orden de convergencia (log-log fit)
    from scipy.stats import linregress
    log_T = np.log(T_values)
    log_rms = np.log(rms_errors)
    slope, intercept, _, _, _ = linregress(log_T, log_rms)
    order = slope

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.loglog(T_values, rms_errors, 'bo-', linewidth=2, markersize=8,
              label='Error RMS', zorder=3)
    ax.loglog(T_values, max_errors, 'rs--', linewidth=2, markersize=8,
              label='Error Máximo', zorder=3)

    # Línea de referencia con pendiente estimada
    T_fit = np.array([T_values[0], T_values[-1]])
    error_fit = np.exp(intercept) * T_fit**order
    ax.loglog(T_fit, error_fit, 'k:', linewidth=2,
              label=f'$O(T^{{{order:.2f}}})$', zorder=2)

    ax.set_xlabel('Paso temporal $T$')
    ax.set_ylabel('Error')
    ax.set_title("Caso 1: Convergencia del Regresor Homotópico", fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)

    # Añadir anotaciones
    ax.text(0.05, 0.05, f'Orden de convergencia: $p \\approx {order:.2f}$',
           transform=ax.transAxes, fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig('caso1_convergence.png', dpi=300, bbox_inches='tight')
    print("  ✓ caso1_convergence.png")
    plt.close()


def figure_3_multiple_n():
    """
    Figura 3: Comparación para diferentes valores de N en subplots.
    """
    print("Generando Figura 3: Comparación múltiple N...")

    y0 = -0.2
    t_span = (0, 10)
    n_values = [100, 500, 2000]

    fig, axes = plt.subplots(len(n_values), 1, figsize=(10, 10))

    for idx, n in enumerate(n_values):
        t_rk4, y_rk4 = solve_ode_rk4(y0, t_span, n)
        t_reg, y_reg = solve_ode_regressor(y0, t_span, n)
        errors = compute_error(y_rk4, y_reg)

        axes[idx].plot(t_rk4, y_rk4, 'b-', linewidth=2, label='odeint')
        axes[idx].plot(t_reg, y_reg, 'ro', markersize=2, label='Regresor', alpha=0.6)
        axes[idx].set_xlabel('Tiempo $t$')
        axes[idx].set_ylabel('$y(t)$')
        axes[idx].set_title(f"$N = {n}$ | Error RMS = ${errors['rms_error']:.2e}$")
        axes[idx].legend(loc='best')
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle("Caso 1: Efecto del número de puntos $N$", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('caso1_multiple_N.png', dpi=300, bbox_inches='tight')
    print("  ✓ caso1_multiple_N.png")
    plt.close()


def figure_4_phase_portrait():
    """
    Figura 4: Retrato de fase y(t) vs y'(t).
    """
    print("Generando Figura 4: Retrato de fase...")

    y0 = -0.2
    t_span = (0, 10)
    n = 2000

    # Soluciones
    t_rk4, y_rk4 = solve_ode_rk4(y0, t_span, n)
    t_reg, y_reg = solve_ode_regressor(y0, t_span, n)

    # Calcular derivadas (aproximación por diferencias finitas)
    dydt_rk4 = np.gradient(y_rk4, t_rk4)
    dydt_reg = np.gradient(y_reg, t_reg)

    # Crear figura
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(y_rk4, dydt_rk4, 'b-', linewidth=2, label='odeint', alpha=0.7)
    ax.plot(y_reg, dydt_reg, 'r--', linewidth=1.5, label='Regresor', alpha=0.8)
    ax.plot(y0, dydt_rk4[0], 'go', markersize=10, label='Condición inicial', zorder=5)

    ax.set_xlabel('$y$')
    ax.set_ylabel("$y'$")
    ax.set_title("Caso 1: Retrato de fase", fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('caso1_phase_portrait.png', dpi=300, bbox_inches='tight')
    print("  ✓ caso1_phase_portrait.png")
    plt.close()


# ============================================================
# Generar todas las figuras
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Generando todas las figuras para Caso 1")
    print("=" * 70)
    print()

    figure_1_comparison()
    figure_2_convergence()
    figure_3_multiple_n()
    figure_4_phase_portrait()

    print()
    print("=" * 70)
    print("Todas las figuras generadas exitosamente ✓")
    print("=" * 70)
    print("\nArchivos generados:")
    print("  - caso1_comparison_N500.png")
    print("  - caso1_convergence.png")
    print("  - caso1_multiple_N.png")
    print("  - caso1_phase_portrait.png")
