"""
Optimización de Parámetros RBF usando Regresor Homotópico

Compara con método tradicional (despeje) para diferentes N puntos.

Author: PublicationE - Marzo 2026
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import sys
sys.path.append('/home/rodo/1Paper/PublicationE')
from rbf_analytical import RBFAnalytical
from duffing_regressor_rbf import solve_duffing_regressor, solve_with_rk45, true_spring_force
import time


# Parámetros del Duffing
D = 0.2
A_SPRING = 1.0
B_SPRING = 0.5
A_FORCE = 0.8
OMEGA = 1.2


def method_traditional(t_data, y_data):
    """
    Método tradicional: Despeje + RBF

    Calcula derivadas numéricas y despeja f(y)
    """
    print(f"\n  {'─'*66}")
    print(f"  MÉTODO 1: Despeje + RBF (Tradicional)")
    print(f"  {'─'*66}")

    start = time.time()

    # Derivadas numéricas
    y_prime = np.gradient(y_data, t_data)
    y_second = np.gradient(y_prime, t_data)

    # Forzamiento
    u = A_FORCE * np.cos(OMEGA * t_data)

    # Despejar: f(y) = u - y'' - d·y'
    f_data = u - y_second - D * y_prime

    # Entrenar RBF
    n_centers = min(8, len(t_data) // 3)
    y_min, y_max = y_data.min(), y_data.max()
    centers = np.linspace(y_min - 0.1, y_max + 0.1, n_centers)
    sigma = (y_max - y_min) / (2 * n_centers)

    distances = cdist(y_data.reshape(-1, 1), centers.reshape(-1, 1))
    phi = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    phi_aug = np.hstack([phi, np.ones((len(y_data), 1))])
    weights = np.linalg.lstsq(phi_aug, f_data, rcond=None)[0]

    rbf = RBFAnalytical(centers, sigma, weights)

    elapsed = time.time() - start

    # Validar con regresor
    y_pred = solve_duffing_regressor(rbf, t_data, y_data[0], y_data[1])

    # Errores
    error_y = np.sqrt(np.mean((y_pred - y_data) ** 2))

    # Error en f(y)
    f_true = true_spring_force(y_data)
    f_rbf = rbf.eval(y_data)
    error_f = np.sqrt(np.mean((f_rbf - f_true) ** 2))

    print(f"    Centros RBF: {n_centers}")
    print(f"    Tiempo: {elapsed:.4f} s")
    print(f"    Error RMSE y(t): {error_y:.6e}")
    print(f"    Error RMSE f(y): {error_f:.6e}")

    return rbf, error_y, error_f, elapsed


def method_optimization(t_data, y_data, n_centers=6):
    """
    Método optimización: Optimizar parámetros RBF directamente

    Minimiza ||y_regresor - y_data||² usando el regresor homotópico
    """
    print(f"\n  {'─'*66}")
    print(f"  MÉTODO 2: Optimización Directa con Regresor")
    print(f"  {'─'*66}")

    start_total = time.time()

    def objective(params):
        """
        Función objetivo: error del regresor

        params = [c1, ..., cM, sigma, w1, ..., wM, w0]
        """
        # Crear RBF con estos parámetros
        rbf_temp = RBFAnalytical(
            centers=params[:n_centers],
            sigma=params[n_centers],
            weights=params[n_centers+1:]
        )

        # Resolver con regresor
        try:
            y_pred = solve_duffing_regressor(rbf_temp, t_data, y_data[0], y_data[1])

            # Error cuadrático medio
            error = np.mean((y_pred - y_data) ** 2)

            return error
        except:
            return 1e10

    # Inicialización
    y_min, y_max = y_data.min(), y_data.max()
    centers_init = np.linspace(y_min - 0.2, y_max + 0.2, n_centers)
    sigma_init = (y_max - y_min) / (2 * n_centers)
    weights_init = np.random.randn(n_centers + 1) * 0.1

    params_init = np.concatenate([centers_init, [sigma_init], weights_init])

    # Bounds
    bounds = []
    for _ in range(n_centers):
        bounds.append((y_min - 0.5, y_max + 0.5))
    bounds.append((0.05, 2.0))
    for _ in range(n_centers + 1):
        bounds.append((-10, 10))

    print(f"    Centros RBF: {n_centers}")
    print(f"    Parámetros totales: {len(params_init)}")
    print(f"    Optimizando...")

    # Optimizar
    result = minimize(
        objective,
        params_init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': False}
    )

    elapsed = time.time() - start_total

    # Crear RBF óptima
    rbf_opt = RBFAnalytical(
        centers=result.x[:n_centers],
        sigma=result.x[n_centers],
        weights=result.x[n_centers+1:]
    )

    # Evaluar
    y_pred = solve_duffing_regressor(rbf_opt, t_data, y_data[0], y_data[1])
    error_y = np.sqrt(np.mean((y_pred - y_data) ** 2))

    # Error en f(y)
    f_true = true_spring_force(y_data)
    f_rbf = rbf_opt.eval(y_data)
    error_f = np.sqrt(np.mean((f_rbf - f_true) ** 2))

    print(f"    Convergió: {result.success}")
    print(f"    Iteraciones: {result.nit}")
    print(f"    Evaluaciones: {result.nfev}")
    print(f"    Tiempo: {elapsed:.2f} s")
    print(f"    Error RMSE y(t): {error_y:.6e}")
    print(f"    Error RMSE f(y): {error_f:.6e}")

    return rbf_opt, error_y, error_f, elapsed, result


def run_comparison(n_points):
    """
    Ejecutar comparación para N puntos
    """
    print(f"\n{'='*70}")
    print(f"COMPARACIÓN CON N = {n_points} PUNTOS")
    print(f"{'='*70}")

    # Generar datos
    t_span = (0, 15)
    y0 = 0.5
    v0 = 0.0

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    t_data, y_data, v_data = solve_with_rk45(t_span, y0, v0, t_eval)

    T = t_data[1] - t_data[0]

    print(f"\nDatos experimentales:")
    print(f"  N puntos: {len(t_data)}")
    print(f"  Paso T: {T:.6f} s")
    print(f"  Rango y: [{y_data.min():.4f}, {y_data.max():.4f}]")

    # Método 1: Tradicional
    rbf_trad, err_y_trad, err_f_trad, time_trad = method_traditional(t_data, y_data)

    # Método 2: Optimización
    n_centers = min(6, n_points // 4)
    rbf_opt, err_y_opt, err_f_opt, time_opt, result_opt = method_optimization(
        t_data, y_data, n_centers=n_centers
    )

    # Calcular mejora
    mejora_y = ((err_y_trad - err_y_opt) / err_y_trad) * 100
    mejora_f = ((err_f_trad - err_f_opt) / err_f_trad) * 100

    # Resumen
    print(f"\n  {'='*66}")
    print(f"  RESUMEN")
    print(f"  {'='*66}")
    print(f"  {'Método':<30} {'RMSE y(t)':<15} {'RMSE f(y)':<15} {'Tiempo'}")
    print(f"  {'-'*66}")
    print(f"  {'Despeje + RBF':<30} {err_y_trad:<15.6e} {err_f_trad:<15.6e} {time_trad:.3f}s")
    print(f"  {'Optimización + Regresor':<30} {err_y_opt:<15.6e} {err_f_opt:<15.6e} {time_opt:.2f}s")
    print(f"  {'-'*66}")
    print(f"  {'Mejora':<30} {mejora_y:>14.1f}% {mejora_f:>14.1f}%")
    print(f"  {'='*66}")

    return {
        'n_points': n_points,
        'T': T,
        'err_y_trad': err_y_trad,
        'err_f_trad': err_f_trad,
        'time_trad': time_trad,
        'err_y_opt': err_y_opt,
        'err_f_opt': err_f_opt,
        'time_opt': time_opt,
        'mejora_y': mejora_y,
        'mejora_f': mejora_f,
        'converged': result_opt.success
    }


def main():
    """Programa principal"""
    print("\n" + "🔵"*35)
    print("OPTIMIZACIÓN DE RBF CON REGRESOR HOMOTÓPICO")
    print("Oscilador de Duffing - Análisis de Sensibilidad")
    print("🔵"*35)

    print(f"\nEcuación: y'' + {D}·y' + f(y) = {A_FORCE}·cos({OMEGA}·t)")
    print(f"Función real: f(y) = {A_SPRING}·y + {B_SPRING}·y³")

    # Probar con diferentes números de puntos
    n_points_list = [20, 30, 40, 50]

    results = []

    for n_points in n_points_list:
        result = run_comparison(n_points)
        results.append(result)

    # Tabla final
    print(f"\n\n{'='*80}")
    print("TABLA RESUMEN FINAL")
    print(f"{'='*80}")
    print(f"{'N':<5} {'T':<10} {'Trad y(t)':<12} {'Opt y(t)':<12} {'Mejora y':<10} "
          f"{'Trad f(y)':<12} {'Opt f(y)':<12} {'Mejora f':<10} {'Conv':<5}")
    print("-"*80)

    for r in results:
        conv_str = '✓' if r['converged'] else '✗'
        print(f"{r['n_points']:<5} {r['T']:<10.6f} {r['err_y_trad']:<12.6e} "
              f"{r['err_y_opt']:<12.6e} {r['mejora_y']:>9.1f}% "
              f"{r['err_f_trad']:<12.6e} {r['err_f_opt']:<12.6e} "
              f"{r['mejora_f']:>9.1f}% {conv_str:<5}")

    print("="*80)

    # Estadísticas
    mejoras_y = [r['mejora_y'] for r in results]
    mejoras_f = [r['mejora_f'] for r in results]

    print(f"\n📊 ESTADÍSTICAS:")
    print(f"  Mejora promedio en y(t): {np.mean(mejoras_y):.1f}%")
    print(f"  Mejora promedio en f(y): {np.mean(mejoras_f):.1f}%")
    print(f"  Mejor mejora en y(t): {np.max(mejoras_y):.1f}% (N={results[np.argmax(mejoras_y)]['n_points']})")
    print(f"  Mejor mejora en f(y): {np.max(mejoras_f):.1f}% (N={results[np.argmax(mejoras_f)]['n_points']})")

    convergidos = sum(1 for r in results if r['converged'])
    print(f"\n  Convergencia: {convergidos}/{len(results)} casos")

    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")

    if np.mean(mejoras_y) > 50:
        print(f"  ✓ Optimización con regresor es SUPERIOR para datos escasos")
        print(f"  ✓ Mejora promedio: {np.mean(mejoras_y):.0f}% en y(t)")
    elif np.mean(mejoras_y) > 0:
        print(f"  ✓ Optimización con regresor mejora resultados")
        print(f"  ✓ Mejora promedio: {np.mean(mejoras_y):.0f}% en y(t)")
    else:
        print(f"  ⚠ Método tradicional es comparable o mejor")

    if convergidos == len(results):
        print(f"  ✓ Convergencia robusta en todos los casos")
    else:
        print(f"  ⚠ Convergencia parcial ({convergidos}/{len(results)} casos)")

    print("\n" + "="*80)
    print("✓ ANÁLISIS COMPLETADO")
    print("="*80 + "\n")


if __name__ == "__main__":
    np.random.seed(42)
    main()
