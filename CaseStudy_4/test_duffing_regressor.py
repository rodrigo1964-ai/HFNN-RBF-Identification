"""
TEST: Regresor Homotópico para Oscilador de Duffing

Compara el regresor vs RK4 para diferentes pasos temporales.

Ecuación: y'' + d·y' + (a·y + b·y³) = A·cos(ω·t)

Author: Publicación E - Marzo 2026
"""
import numpy as np
from scipy.integrate import solve_ivp
import sys
sys.path.append('/home/rodo/regressor')
from regressor import build_regressor_order2
from sympy import Symbol
import time

# Parámetros del oscilador de Duffing
D = 0.2          # Amortiguamiento
A_SPRING = 1.0   # Coeficiente lineal
B_SPRING = 0.5   # Coeficiente cúbico
A_FORCE = 0.8    # Amplitud del forzamiento
OMEGA = 1.2      # Frecuencia del forzamiento


def duffing_ode(t, state):
    """Ecuación del Duffing para RK45"""
    y, v = state
    dy_dt = v
    dv_dt = A_FORCE * np.cos(OMEGA * t) - D * v - (A_SPRING * y + B_SPRING * y**3)
    return [dy_dt, dv_dt]


def solve_with_rk45(t_span, y0, v0, n_points):
    """Resolver con RK45 (referencia de alta precisión)"""
    print(f"\n{'='*70}")
    print("RESOLVIENDO CON RK45 (REFERENCIA)")
    print(f"{'='*70}")

    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    start = time.time()
    sol = solve_ivp(
        duffing_ode,
        t_span,
        [y0, v0],
        t_eval=t_eval,
        method='RK45',
        rtol=1e-10,
        atol=1e-12
    )
    elapsed = time.time() - start

    print(f"  Puntos: {n_points}")
    print(f"  Tiempo: {elapsed:.4f} s")
    print(f"  Éxito: {sol.success}")

    return sol.t, sol.y[0], sol.y[1]


def solve_with_regressor(t_data, y_rk4, v_rk4, T):
    """Resolver con regresor homotópico"""
    print(f"\n{'='*70}")
    print(f"RESOLVIENDO CON REGRESOR HOMOTÓPICO (T = {T:.6f})")
    print(f"{'='*70}")

    # Construir el regresor simbólicamente
    # f(y, y') = d·y' + a·y + b·y³
    y_sym = Symbol('y')
    yp_sym = Symbol('yp')

    f_expr = D * yp_sym + A_SPRING * y_sym + B_SPRING * y_sym**3

    print("\nGenerando regresor simbólico...")
    regressor, info = build_regressor_order2(f_expr, y_sym, yp_sym)

    # Forzamiento
    u = A_FORCE * np.cos(OMEGA * t_data)

    # Resolver con regresor
    print(f"\nResolviendo con regresor...")
    start = time.time()
    y_reg = regressor(u, y_rk4[0], y_rk4[1], T, len(t_data))
    elapsed = time.time() - start

    # Calcular errores
    error_abs = np.abs(y_reg - y_rk4)
    error_max = np.max(error_abs)
    error_rms = np.sqrt(np.mean(error_abs**2))

    print(f"\n  Resultados:")
    print(f"    Tiempo: {elapsed:.4f} s")
    print(f"    Error máximo: {error_max:.6e}")
    print(f"    Error RMS: {error_rms:.6e}")

    return y_reg, error_max, error_rms, elapsed


def test_different_step_sizes():
    """Probar diferentes pasos temporales"""
    print("\n" + "="*70)
    print("TEST: OSCILADOR DE DUFFING CON REGRESOR HOMOTÓPICO")
    print("="*70)
    print(f"\nParámetros del sistema:")
    print(f"  y'' + {D}·y' + ({A_SPRING}·y + {B_SPRING}·y³) = {A_FORCE}·cos({OMEGA}·t)")

    # Configuración
    t_span = (0, 15)
    y0 = 0.5
    v0 = 0.0

    # Probar diferentes números de puntos (diferentes pasos T)
    n_points_list = [100, 200, 500, 1000, 2000, 5000]

    results = []

    for n_points in n_points_list:
        print(f"\n{'#'*70}")
        print(f"# PRUEBA CON N = {n_points} PUNTOS")
        print(f"{'#'*70}")

        # Resolver con RK45
        t_rk4, y_rk4, v_rk4 = solve_with_rk45(t_span, y0, v0, n_points)

        # Paso temporal
        T = t_rk4[1] - t_rk4[0]

        # Resolver con regresor
        y_reg, err_max, err_rms, time_reg = solve_with_regressor(
            t_rk4, y_rk4, v_rk4, T
        )

        results.append({
            'n_points': n_points,
            'T': T,
            'error_max': err_max,
            'error_rms': err_rms,
            'time': time_reg
        })

    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN DE RESULTADOS")
    print(f"{'='*70}")
    print(f"\n{'N Puntos':<10} {'T (paso)':<12} {'Error Max':<15} {'Error RMS':<15} {'Tiempo':<10}")
    print("-"*70)

    for r in results:
        print(f"{r['n_points']:<10} {r['T']:<12.6f} {r['error_max']:<15.6e} "
              f"{r['error_rms']:<15.6e} {r['time']:<10.4f}s")

    # Encontrar mejor configuración
    best = min(results, key=lambda x: x['error_max'])
    print(f"\n{'='*70}")
    print("MEJOR CONFIGURACIÓN:")
    print(f"  N = {best['n_points']} puntos")
    print(f"  T = {best['T']:.6f} s")
    print(f"  Error máximo = {best['error_max']:.6e}")
    print(f"  Error RMS = {best['error_rms']:.6e}")
    print(f"{'='*70}")

    # Recomendación
    print(f"\n💡 RECOMENDACIÓN:")
    acceptable = [r for r in results if r['error_max'] < 1e-2]
    if acceptable:
        best_acceptable = min(acceptable, key=lambda x: x['time'])
        print(f"   Para error < 1e-2:")
        print(f"     • Usar N ≥ {best_acceptable['n_points']} puntos")
        print(f"     • T ≤ {best_acceptable['T']:.6f} s")
        print(f"     • Error esperado: {best_acceptable['error_max']:.2e}")

    precision = [r for r in results if r['error_max'] < 1e-3]
    if precision:
        best_precision = min(precision, key=lambda x: x['time'])
        print(f"\n   Para alta precisión (error < 1e-3):")
        print(f"     • Usar N ≥ {best_precision['n_points']} puntos")
        print(f"     • T ≤ {best_precision['T']:.6f} s")
        print(f"     • Error esperado: {best_precision['error_max']:.2e}")

    return results


def test_single_configuration():
    """Prueba detallada con configuración específica"""
    print("\n" + "="*70)
    print("PRUEBA DETALLADA CON CONFIGURACIÓN RECOMENDADA")
    print("="*70)

    t_span = (0, 15)
    y0 = 0.5
    v0 = 0.0
    n_points = 2000  # Configuración esperada óptima

    # Resolver con RK45
    t_rk4, y_rk4, v_rk4 = solve_with_rk45(t_span, y0, v0, n_points)
    T = t_rk4[1] - t_rk4[0]

    # Resolver con regresor
    y_reg, err_max, err_rms, time_reg = solve_with_regressor(
        t_rk4, y_rk4, v_rk4, T
    )

    # Comparación punto por punto (primeros 10 y últimos 10)
    print(f"\n{'='*70}")
    print("COMPARACIÓN PUNTO POR PUNTO (primeros 10 puntos):")
    print(f"{'='*70}")
    print(f"{'i':<5} {'t':<10} {'y_RK4':<15} {'y_Regresor':<15} {'Error':<15}")
    print("-"*70)
    for i in range(10):
        print(f"{i:<5} {t_rk4[i]:<10.4f} {y_rk4[i]:<15.8f} "
              f"{y_reg[i]:<15.8f} {np.abs(y_reg[i] - y_rk4[i]):<15.6e}")

    print(f"\n{'='*70}")
    print("COMPARACIÓN PUNTO POR PUNTO (últimos 10 puntos):")
    print(f"{'='*70}")
    print(f"{'i':<5} {'t':<10} {'y_RK4':<15} {'y_Regresor':<15} {'Error':<15}")
    print("-"*70)
    for i in range(-10, 0):
        idx = len(t_rk4) + i
        print(f"{idx:<5} {t_rk4[i]:<10.4f} {y_rk4[i]:<15.8f} "
              f"{y_reg[i]:<15.8f} {np.abs(y_reg[i] - y_rk4[i]):<15.6e}")

    print(f"\n{'='*70}")
    print("✓ VERIFICACIÓN COMPLETADA")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Ejecutar tests
    results = test_different_step_sizes()

    print("\n\n")
    test_single_configuration()

    print("\n\n✓ Todos los tests completados")
