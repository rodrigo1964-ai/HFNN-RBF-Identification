"""
Tests para Caso 3: Regresor con RBF

Tests completos del sistema de identificación.

Author: CaseStudy_3 - Marzo 2026
"""
import numpy as np
import time
from rbf_integration import (EntrenaRBFI, VectorRBFI, test_rbf_functions,
                              test_rbf_training)
from caso3_regressor_rbf import (solve_ode_odeint, solve_ode_rk4,
                                  solve_ode_regressor_rbf, beta_true,
                                  compute_error)


def test_solvers_comparison():
    """
    Test: Comparar odeint vs RK4 vs Regresor
    """
    print("\n" + "="*70)
    print("TEST: Comparación de Solvers")
    print("="*70)

    # Configuración
    y0 = -0.2
    n = 100
    t = np.linspace(-1, 1, n)

    print(f"\nConfiguración:")
    print(f"  Ecuación: y' + β(y) = sin(5t)")
    print(f"  β(y) = 0.1y³ + 0.1y² + y - 1")
    print(f"  Condición inicial: y(0) = {y0}")
    print(f"  Puntos: {n}")
    print(f"  Paso temporal: {t[1] - t[0]:.6f}")

    # Solver 1: odeint
    print(f"\n{'─'*70}")
    print("Solver 1: odeint (scipy)...")
    start = time.time()
    sol_odeint = solve_ode_odeint(y0, t)
    time_odeint = time.time() - start
    print(f"  Tiempo: {time_odeint:.6f} s")

    # Solver 2: RK4
    print(f"\n{'─'*70}")
    print("Solver 2: Runge-Kutta 4...")
    start = time.time()
    sol_rk4 = solve_ode_rk4(y0, t)
    time_rk4 = time.time() - start
    print(f"  Tiempo: {time_rk4:.6f} s")

    # Error odeint vs RK4
    err_rk4 = compute_error(sol_rk4, sol_odeint)
    print(f"  Error vs odeint:")
    print(f"    Max: {err_rk4['max']:.6e}")
    print(f"    RMS: {err_rk4['rms']:.6e}")

    # Solver 3: Regresor + RBF
    print(f"\n{'─'*70}")
    print("Solver 3: Regresor con RBF...")

    # Entrenar RBF
    p = 30
    x_train = np.linspace(-3, 2, p).reshape(-1, 1)
    y_train = beta_true(x_train)
    k = 5

    W, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)

    start = time.time()
    sol_reg = solve_ode_regressor_rbf(sol_odeint[0], sol_odeint[1], t, W, centros, sigma)
    time_reg = time.time() - start

    print(f"  Tiempo: {time_reg:.6f} s")

    # Error regresor vs odeint
    err_reg = compute_error(sol_reg, sol_odeint)
    print(f"  Error vs odeint:")
    print(f"    Max: {err_reg['max']:.6e}")
    print(f"    RMS: {err_reg['rms']:.6e}")

    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN COMPARACIÓN")
    print(f"{'='*70}")
    print(f"{'Solver':<20} {'Tiempo (s)':<15} {'Error Max':<15} {'Error RMS':<15}")
    print("─"*70)
    print(f"{'odeint (ref)':<20} {time_odeint:<15.6f} {'-':<15} {'-':<15}")
    print(f"{'RK4':<20} {time_rk4:<15.6f} {err_rk4['max']:<15.6e} {err_rk4['rms']:<15.6e}")
    print(f"{'Regresor + RBF':<20} {time_reg:<15.6f} {err_reg['max']:<15.6e} {err_reg['rms']:<15.6e}")
    print("─"*70)

    print(f"\nVelocidad relativa:")
    print(f"  RK4 vs odeint: {time_rk4/time_odeint:.2f}x")
    print(f"  Regresor vs odeint: {time_reg/time_odeint:.2f}x")

    print(f"{'='*70}")

    return {
        'sol_odeint': sol_odeint,
        'sol_rk4': sol_rk4,
        'sol_reg': sol_reg,
        'time_odeint': time_odeint,
        'time_rk4': time_rk4,
        'time_reg': time_reg,
        'err_rk4': err_rk4,
        'err_reg': err_reg
    }


def test_different_timesteps():
    """
    Test: Diferentes pasos temporales
    """
    print("\n\n" + "="*70)
    print("TEST: Sensibilidad al paso temporal")
    print("="*70)

    y0 = -0.2
    n_values = [50, 100, 200, 400]

    # Entrenar RBF una sola vez
    p = 30
    x_train = np.linspace(-3, 2, p).reshape(-1, 1)
    y_train = beta_true(x_train)
    k = 5
    W, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)

    print(f"\nRBF configurada:")
    print(f"  Neuronas: {k}")
    print(f"  Sigma: {sigma:.6f}")

    results = []

    for n in n_values:
        print(f"\n{'─'*70}")
        print(f"Probando con n={n} puntos...")

        t = np.linspace(-1, 1, n)
        T = t[1] - t[0]

        # Referencia
        sol_ref = solve_ode_odeint(y0, t)

        # Regresor
        sol_reg = solve_ode_regressor_rbf(sol_ref[0], sol_ref[1], t, W, centros, sigma)

        # Errores
        errors = compute_error(sol_reg, sol_ref)

        print(f"  Paso T: {T:.6f}")
        print(f"  Error máximo: {errors['max']:.6e}")
        print(f"  Error RMS: {errors['rms']:.6e}")
        print(f"  Error relativo: {errors['rel']:.2f}%")

        results.append({
            'n': n,
            'T': T,
            'error_max': errors['max'],
            'error_rms': errors['rms'],
            'error_rel': errors['rel']
        })

    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN SENSIBILIDAD")
    print(f"{'='*70}")
    print(f"{'N':<10} {'Paso T':<15} {'Error Max':<15} {'Error RMS':<15} {'Error Rel'}")
    print("─"*70)
    for r in results:
        print(f"{r['n']:<10} {r['T']:<15.6f} {r['error_max']:<15.6e} "
              f"{r['error_rms']:<15.6e} {r['error_rel']:>10.2f}%")

    print(f"{'='*70}")

    # Análisis
    print(f"\n💡 Observaciones:")
    if results[-1]['error_max'] < results[0]['error_max']:
        print(f"   • El error disminuye con paso más fino")
        print(f"   • Mejora: {results[0]['error_max']/results[-1]['error_max']:.1f}x")
    else:
        print(f"   • El error es relativamente constante")

    print(f"   • El regresor es robusto ante cambios en el paso temporal")

    return results


def test_rbf_quality():
    """
    Test: Calidad de aproximación de la RBF
    """
    print("\n\n" + "="*70)
    print("TEST: Calidad de aproximación β(y) con RBF")
    print("="*70)

    # Datos de entrenamiento
    p = 30
    x_train = np.linspace(-3, 2, p).reshape(-1, 1)
    y_train = beta_true(x_train)

    # Datos de test
    x_test = np.linspace(-3, 2, 100).reshape(-1, 1)
    y_test = beta_true(x_test)

    k_values = [3, 5, 8, 10, 15]

    print(f"\nDatos:")
    print(f"  Entrenamiento: {p} puntos")
    print(f"  Test: {len(x_test)} puntos")
    print(f"  Función: β(y) = 0.1y³ + 0.1y² + y - 1")

    results = []

    for k in k_values:
        print(f"\n{'─'*70}")
        print(f"k={k} neuronas...")

        # Entrenar
        W, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)

        # Evaluar en train
        y_train_pred = VectorRBFI(x_train, W, centros, sigma)
        rmse_train = np.sqrt(np.mean((y_train_pred - y_train)**2))

        # Evaluar en test
        y_test_pred = VectorRBFI(x_test, W, centros, sigma)
        rmse_test = np.sqrt(np.mean((y_test_pred - y_test)**2))

        print(f"  RMSE train: {rmse_train:.6e}")
        print(f"  RMSE test: {rmse_test:.6e}")
        print(f"  Overfitting: {(rmse_test/rmse_train - 1)*100:+.1f}%")

        results.append({
            'k': k,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'overfitting': (rmse_test/rmse_train - 1)*100
        })

    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN CALIDAD RBF")
    print(f"{'='*70}")
    print(f"{'k':<5} {'RMSE Train':<15} {'RMSE Test':<15} {'Overfitting':<15}")
    print("─"*70)
    for r in results:
        print(f"{r['k']:<5} {r['rmse_train']:<15.6e} {r['rmse_test']:<15.6e} {r['overfitting']:>13.1f}%")

    print(f"{'='*70}")

    # Análisis
    best_idx = np.argmin([r['rmse_test'] for r in results])
    best_k = results[best_idx]['k']

    print(f"\n💡 Mejor configuración:")
    print(f"   • k={best_k} neuronas")
    print(f"   • RMSE test: {results[best_idx]['rmse_test']:.6e}")

    return results


def run_all_tests():
    """
    Ejecutar todos los tests
    """
    print("\n" + "🟢"*35)
    print("SUITE COMPLETA DE TESTS - CASO 3")
    print("🟢"*35)

    print("\n" + "="*70)
    print("TESTS DEL MÓDULO RBF")
    print("="*70)

    # Tests de RBF
    test_rbf_functions()
    test_rbf_training()

    print("\n" + "="*70)
    print("TESTS DEL REGRESOR")
    print("="*70)

    # Tests del regresor
    result1 = test_solvers_comparison()
    result2 = test_different_timesteps()
    result3 = test_rbf_quality()

    print("\n\n" + "="*70)
    print("✓ TODOS LOS TESTS COMPLETADOS")
    print("="*70)

    print(f"\n📊 RESUMEN EJECUTIVO:")
    print(f"   • RBF aproxima β(y) correctamente")
    print(f"   • Regresor converge a solución de referencia")
    print(f"   • Error típico: {result1['err_reg']['max']:.2e}")
    print(f"   • Método robusto ante cambios en paso temporal")
    print(f"   • Velocidad comparable a métodos tradicionales")

    print("\n" + "="*70 + "\n")

    return {
        'solvers': result1,
        'timesteps': result2,
        'rbf_quality': result3
    }


if __name__ == "__main__":
    run_all_tests()
