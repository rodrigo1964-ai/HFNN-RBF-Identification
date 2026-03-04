"""
test_caso1.py - Tests y comparaciones para el Caso 1

Compara el regresor homotópico con odeint para diferentes valores de N.

Author: Rodolfo H. Rodrigo - UNSJ
Fecha: Marzo 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from caso1_regressor import (
    solve_ode_regressor, solve_ode_rk4, compute_error
)


def test_multiple_n():
    """
    Prueba el regresor con diferentes números de puntos.
    Analiza la convergencia del método.
    """
    print("=" * 70)
    print("Test de Convergencia: Variando N")
    print("=" * 70)

    y0 = -0.2
    t_span = (0, 10)
    n_values = [100, 200, 500, 1000, 2000, 5000]

    results = []

    for n in n_values:
        # Solución de referencia
        t_rk4, y_rk4 = solve_ode_rk4(y0, t_span, n)

        # Solución con regresor
        t_reg, y_reg = solve_ode_regressor(y0, t_span, n)

        # Calcular errores
        errors = compute_error(y_rk4, y_reg)
        T = t_rk4[1] - t_rk4[0]

        results.append({
            'n': n,
            'T': T,
            'max_error': errors['max_error'],
            'rms_error': errors['rms_error'],
            'rel_error': errors['rel_error']
        })

        print(f"N={n:5d} | T={T:.6f} | Error máx={errors['max_error']:.4e} | "
              f"RMS={errors['rms_error']:.4e} | Rel={errors['rel_error']:.2f}%")

    print("=" * 70)

    return results


def test_comparison_plot():
    """
    Genera gráfico de comparación entre regresor y odeint.
    """
    print("\n" + "=" * 70)
    print("Generando gráfico de comparación...")
    print("=" * 70)

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
    axes[0].plot(t_rk4, y_rk4, 'b-', linewidth=2, label='odeint (RK4)')
    axes[0].plot(t_reg, y_reg, 'ro', markersize=3, label='Regresor Homotópico')
    axes[0].set_xlabel('Tiempo t', fontsize=12)
    axes[0].set_ylabel('y(t)', fontsize=12)
    axes[0].set_title(f"Caso 1: y' + y² = sin(5t) | N={n}", fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Error absoluto
    axes[1].plot(t_rk4, error_abs, 'r-', linewidth=1.5)
    axes[1].set_xlabel('Tiempo t', fontsize=12)
    axes[1].set_ylabel('Error absoluto', fontsize=12)
    axes[1].set_title(f"Error: máx={errors['max_error']:.4e}, RMS={errors['rms_error']:.4e}",
                     fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('caso1_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: caso1_comparison.png")
    plt.show()


def test_convergence_plot(results):
    """
    Genera gráfico de convergencia (Error vs T).

    Parameters
    ----------
    results : list
        Lista de resultados del test_multiple_n()
    """
    print("\n" + "=" * 70)
    print("Generando gráfico de convergencia...")
    print("=" * 70)

    T_values = [r['T'] for r in results]
    rms_errors = [r['rms_error'] for r in results]
    max_errors = [r['max_error'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(T_values, rms_errors, 'bo-', linewidth=2, markersize=8,
              label='Error RMS')
    ax.loglog(T_values, max_errors, 'rs--', linewidth=2, markersize=8,
              label='Error Máximo')

    # Línea de referencia O(T)
    T_ref = np.array(T_values)
    error_ref = rms_errors[0] * (T_ref / T_values[0])
    ax.loglog(T_ref, error_ref, 'k:', linewidth=1.5, label='O(T) referencia')

    ax.set_xlabel('Paso temporal T', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title("Caso 1: Convergencia del Regresor Homotópico",
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('caso1_convergence.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: caso1_convergence.png")
    plt.show()


# ============================================================
# Ejecutar todos los tests
# ============================================================
if __name__ == "__main__":
    # Test de convergencia
    results = test_multiple_n()

    # Gráfico de comparación
    test_comparison_plot()

    # Gráfico de convergencia
    test_convergence_plot(results)

    print("\n" + "=" * 70)
    print("Tests completados exitosamente ✓")
    print("=" * 70)
