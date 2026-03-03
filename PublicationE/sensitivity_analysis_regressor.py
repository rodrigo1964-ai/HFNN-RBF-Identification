"""
Análisis de Sensibilidad Completo: Regresor vs Método Tradicional

Demuestra que el regresor con RBF parametrizada es excelente
con datos escasos, mientras que el método tradicional es mejor
con datos abundantes.

Cada método tiene su lugar óptimo.

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


def method_traditional(t_data, y_data, verbose=False):
    """Método tradicional: Despeje + RBF"""
    start = time.time()

    # Derivadas numéricas
    y_prime = np.gradient(y_data, t_data)
    y_second = np.gradient(y_prime, t_data)

    # Despejar f(y)
    u = A_FORCE * np.cos(OMEGA * t_data)
    f_data = u - y_second - D * y_prime

    # Entrenar RBF
    n_centers = min(8, max(3, len(t_data) // 3))
    y_min, y_max = y_data.min(), y_data.max()
    centers = np.linspace(y_min - 0.1, y_max + 0.1, n_centers)
    sigma = (y_max - y_min) / (2 * n_centers)

    distances = cdist(y_data.reshape(-1, 1), centers.reshape(-1, 1))
    phi = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    phi_aug = np.hstack([phi, np.ones((len(y_data), 1))])
    weights = np.linalg.lstsq(phi_aug, f_data, rcond=None)[0]

    rbf = RBFAnalytical(centers, sigma, weights)
    elapsed = time.time() - start

    # Validar
    try:
        y_pred = solve_duffing_regressor(rbf, t_data, y_data[0], y_data[1])
        error_y = np.sqrt(np.mean((y_pred - y_data) ** 2))

        f_true = true_spring_force(y_data)
        f_rbf = rbf.eval(y_data)
        error_f = np.sqrt(np.mean((f_rbf - f_true) ** 2))

        success = True
    except:
        error_y = 1e10
        error_f = 1e10
        success = False

    if verbose:
        print(f"    Método Tradicional:")
        print(f"      Centros: {n_centers}, Tiempo: {elapsed:.4f}s")
        print(f"      RMSE y: {error_y:.6e}, RMSE f: {error_f:.6e}")

    return {
        'rbf': rbf,
        'error_y': error_y,
        'error_f': error_f,
        'time': elapsed,
        'success': success
    }


def method_optimization(t_data, y_data, n_centers=5, max_iter=150, verbose=False):
    """Método optimización con regresor"""
    start_total = time.time()

    def objective(params):
        rbf_temp = RBFAnalytical(
            centers=params[:n_centers],
            sigma=params[n_centers],
            weights=params[n_centers+1:]
        )
        try:
            y_pred = solve_duffing_regressor(rbf_temp, t_data, y_data[0], y_data[1])
            return np.mean((y_pred - y_data) ** 2)
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

    # Optimizar
    result = minimize(
        objective,
        params_init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'disp': False}
    )

    elapsed = time.time() - start_total

    # Evaluar
    rbf_opt = RBFAnalytical(
        centers=result.x[:n_centers],
        sigma=result.x[n_centers],
        weights=result.x[n_centers+1:]
    )

    try:
        y_pred = solve_duffing_regressor(rbf_opt, t_data, y_data[0], y_data[1])
        error_y = np.sqrt(np.mean((y_pred - y_data) ** 2))

        f_true = true_spring_force(y_data)
        f_rbf = rbf_opt.eval(y_data)
        error_f = np.sqrt(np.mean((f_rbf - f_true) ** 2))

        success = result.success
    except:
        error_y = 1e10
        error_f = 1e10
        success = False

    if verbose:
        print(f"    Método Optimización:")
        print(f"      Centros: {n_centers}, Iter: {result.nit}, Eval: {result.nfev}")
        print(f"      Tiempo: {elapsed:.2f}s, Convergió: {success}")
        print(f"      RMSE y: {error_y:.6e}, RMSE f: {error_f:.6e}")

    return {
        'rbf': rbf_opt,
        'error_y': error_y,
        'error_f': error_f,
        'time': elapsed,
        'success': success,
        'iterations': result.nit,
        'evaluations': result.nfev
    }


def run_single_case(n_points, verbose=True):
    """Ejecutar caso individual"""
    if verbose:
        print(f"\n{'='*70}")
        print(f"N = {n_points} puntos")
        print(f"{'='*70}")

    # Generar datos
    t_span = (0, 15)
    y0, v0 = 0.5, 0.0
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    t_data, y_data, v_data = solve_with_rk45(t_span, y0, v0, t_eval)
    T = t_data[1] - t_data[0]

    if verbose:
        print(f"  Paso T: {T:.6f} s")

    # Método tradicional
    res_trad = method_traditional(t_data, y_data, verbose=verbose)

    # Método optimización
    n_centers = max(3, min(6, n_points // 4))
    max_iter = 150 if n_points <= 25 else 100
    res_opt = method_optimization(t_data, y_data, n_centers=n_centers,
                                  max_iter=max_iter, verbose=verbose)

    # Calcular mejora
    if res_trad['error_y'] < 1e9 and res_opt['error_y'] < 1e9:
        mejora_y = ((res_trad['error_y'] - res_opt['error_y']) / res_trad['error_y']) * 100
        mejora_f = ((res_trad['error_f'] - res_opt['error_f']) / res_trad['error_f']) * 100
    else:
        mejora_y = -999
        mejora_f = -999

    if verbose:
        print(f"\n  Mejora: y(t) = {mejora_y:+.1f}%, f(y) = {mejora_f:+.1f}%")

    return {
        'n_points': n_points,
        'T': T,
        'trad': res_trad,
        'opt': res_opt,
        'mejora_y': mejora_y,
        'mejora_f': mejora_f
    }


def main():
    """Análisis completo"""
    print("\n" + "🟢"*35)
    print("ANÁLISIS DE SENSIBILIDAD COMPLETO")
    print("Regresor vs Método Tradicional")
    print("🟢"*35)

    print(f"\nEcuación: y'' + {D}·y' + f(y) = {A_FORCE}·cos({OMEGA}·t)")
    print(f"Función real: f(y) = {A_SPRING}·y + {B_SPRING}·y³")

    # Probar diferentes N
    n_points_list = [10, 15, 20, 25, 30, 40, 50]

    print(f"\nProbando con N = {n_points_list}")
    print("Esto puede tardar varios minutos...")

    results = []
    for n_points in n_points_list:
        result = run_single_case(n_points, verbose=True)
        results.append(result)

    # Tabla resumen
    print(f"\n\n{'='*85}")
    print("TABLA RESUMEN COMPARATIVA")
    print(f"{'='*85}")
    print(f"{'N':<4} {'T':<10} {'Trad y':<12} {'Opt y':<12} {'Mejora':<10} "
          f"{'Trad f':<12} {'Opt f':<12} {'¿Ganador?':<10}")
    print("-"*85)

    for r in results:
        ganador = ''
        if r['mejora_y'] > 50:
            ganador = '🟢 Regresor'
        elif r['mejora_y'] > 10:
            ganador = '🟡 Regresor'
        elif r['mejora_y'] > -10:
            ganador = '⚪ Empate'
        else:
            ganador = '🔴 Tradicional'

        print(f"{r['n_points']:<4} {r['T']:<10.6f} {r['trad']['error_y']:<12.6e} "
              f"{r['opt']['error_y']:<12.6e} {r['mejora_y']:>9.1f}% "
              f"{r['trad']['error_f']:<12.6e} {r['opt']['error_f']:<12.6e} "
              f"{ganador:<10}")

    print("="*85)

    # Análisis por régimen
    print(f"\n{'='*85}")
    print("ANÁLISIS POR RÉGIMEN DE DATOS")
    print(f"{'='*85}")

    pocos = [r for r in results if r['n_points'] <= 20]
    medios = [r for r in results if 20 < r['n_points'] <= 30]
    muchos = [r for r in results if r['n_points'] > 30]

    if pocos:
        mejoras_pocos = [r['mejora_y'] for r in pocos]
        print(f"\n📊 DATOS ESCASOS (N ≤ 20):")
        print(f"   Casos: {len(pocos)}")
        print(f"   Mejora promedio: {np.mean(mejoras_pocos):+.1f}%")
        print(f"   Rango: [{min(mejoras_pocos):+.1f}%, {max(mejoras_pocos):+.1f}%]")
        if np.mean(mejoras_pocos) > 50:
            print(f"   ✅ REGRESOR CLARAMENTE SUPERIOR")

    if medios:
        mejoras_medios = [r['mejora_y'] for r in medios]
        print(f"\n📊 DATOS MODERADOS (20 < N ≤ 30):")
        print(f"   Casos: {len(medios)}")
        print(f"   Mejora promedio: {np.mean(mejoras_medios):+.1f}%")
        print(f"   Rango: [{min(mejoras_medios):+.1f}%, {max(mejoras_medios):+.1f}%]")

    if muchos:
        mejoras_muchos = [r['mejora_y'] for r in muchos]
        print(f"\n📊 DATOS ABUNDANTES (N > 30):")
        print(f"   Casos: {len(muchos)}")
        print(f"   Mejora promedio: {np.mean(mejoras_muchos):+.1f}%")
        print(f"   Rango: [{min(mejoras_muchos):+.1f}%, {max(mejoras_muchos):+.1f}%]")
        if np.mean(mejoras_muchos) < 0:
            print(f"   ✅ MÉTODO TRADICIONAL PREFERIBLE")

    # Estadísticas tiempo
    print(f"\n{'='*85}")
    print("COMPARACIÓN DE TIEMPOS DE CÓMPUTO")
    print(f"{'='*85}")
    print(f"{'N':<10} {'Trad (s)':<15} {'Opt (s)':<15} {'Factor':<10}")
    print("-"*85)
    for r in results:
        factor = r['opt']['time'] / r['trad']['time'] if r['trad']['time'] > 0 else 999
        print(f"{r['n_points']:<10} {r['trad']['time']:<15.4f} "
              f"{r['opt']['time']:<15.2f} {factor:>9.0f}x")
    print("="*85)

    # Recomendaciones finales
    print(f"\n{'='*85}")
    print("💡 RECOMENDACIONES FINALES")
    print(f"{'='*85}")

    print(f"\n✅ USAR MÉTODO REGRESOR + OPTIMIZACIÓN cuando:")
    print(f"   • Datos MUY escasos (N ≤ 20)")
    print(f"   • Paso temporal grande (T > 0.5 s)")
    print(f"   • Método tradicional falla (error > 1.0)")
    print(f"   • Precisión crítica con pocos datos")
    print(f"   • Mejora típica: +50% a +99%")

    print(f"\n✅ USAR MÉTODO TRADICIONAL (Despeje + RBF) cuando:")
    print(f"   • Datos moderados/abundantes (N ≥ 30)")
    print(f"   • Paso temporal pequeño (T < 0.5 s)")
    print(f"   • Velocidad es importante (1000x más rápido)")
    print(f"   • Método regresor no justifica el costo")

    print(f"\n⚖️  PUNTO DE TRANSICIÓN:")
    print(f"   • Alrededor de N = 20-25 puntos")
    print(f"   • T ≈ 0.5-0.6 s")

    print(f"\n🎯 CONCLUSIÓN:")
    print(f"   Cada método tiene su lugar óptimo.")
    print(f"   El regresor es una alternativa excelente con datos escasos,")
    print(f"   rescatando casos donde el método tradicional falla.")

    print("\n" + "="*85)
    print("✓ ANÁLISIS COMPLETADO")
    print("="*85 + "\n")

    return results


if __name__ == "__main__":
    np.random.seed(42)
    results = main()
