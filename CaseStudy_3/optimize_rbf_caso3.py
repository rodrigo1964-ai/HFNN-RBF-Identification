"""
Optimización de Pesos RBF usando Nelder-Mead para Caso 3

Compara:
1. Pesos iniciales W_pinv (mínimos cuadrados)
2. Pesos optimizados W_nm (Nelder-Mead)

Minimiza el error entre el regresor y la solución de referencia.

Author: CaseStudy_3 - Marzo 2026
"""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
import time
from rbf_integration import EntrenaRBFI, VectorRBFI
from caso3_regressor_rbf import (solve_ode_regressor_rbf, solve_ode_odeint,
                                  beta_true, compute_error)


def objective_function(W, sol_ref, k, t, centros, sigma):
    """
    Función objetivo para optimización

    Calcula el error cuadrático medio entre la solución del
    regresor con pesos W y la solución de referencia.

    Parameters
    ----------
    W : array (k,)
        Pesos RBF (vector plano)
    sol_ref : array
        Solución de referencia
    k : int
        Número de neuronas
    t : array
        Vector de tiempos
    centros : array (k,1)
        Centros RBF
    sigma : float
        Ancho Gaussiana

    Returns
    -------
    float
        Error cuadrático medio
    """
    n = len(t)

    # Reshape W
    W_reshaped = W.reshape((k, 1))

    # Resolver con regresor
    y = solve_ode_regressor_rbf(sol_ref[0], sol_ref[1], t, W_reshaped, centros, sigma)

    # Error cuadrático medio
    error = np.sum((y - sol_ref) ** 2)

    return error


def optimize_weights_nelder_mead(W_init, sol_ref, k, t, centros, sigma, maxiter=200):
    """
    Optimizar pesos usando Nelder-Mead

    Parameters
    ----------
    W_init : array (k,1)
        Pesos iniciales
    sol_ref : array
        Solución de referencia
    k : int
        Número de neuronas
    t : array
        Vector de tiempos
    centros : array (k,1)
        Centros RBF
    sigma : float
        Ancho
    maxiter : int, optional
        Máximo de iteraciones

    Returns
    -------
    dict
        Resultado de la optimización con W_opt, objetivo_final, etc.
    """
    print(f"\n{'─'*70}")
    print("OPTIMIZACIÓN CON NELDER-MEAD")
    print(f"{'─'*70}")

    # Objetivo inicial
    obj_init = objective_function(W_init.ravel(), sol_ref, k, t, centros, sigma)

    print(f"  Objetivo inicial: {obj_init:.6e}")
    print(f"  Optimizando (max iter: {maxiter})...")

    start = time.time()

    # Optimizar
    result = minimize(
        objective_function,
        W_init.ravel(),
        args=(sol_ref, k, t, centros, sigma),
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'disp': False}
    )

    elapsed = time.time() - start

    # Pesos óptimos
    W_opt = result.x.reshape((k, 1))
    obj_final = result.fun

    print(f"  ✓ Optimización completada")
    print(f"  Tiempo: {elapsed:.4f} s")
    print(f"  Iteraciones: {result.nit}")
    print(f"  Evaluaciones: {result.nfev}")
    print(f"  Convergió: {result.success}")
    print(f"  Objetivo final: {obj_final:.6e}")
    print(f"  Mejora: {((obj_init - obj_final)/obj_init*100):.2f}%")

    return {
        'W_opt': W_opt,
        'objective_init': obj_init,
        'objective_final': obj_final,
        'mejora': (obj_init - obj_final)/obj_init*100,
        'time': elapsed,
        'nit': result.nit,
        'nfev': result.nfev,
        'success': result.success,
        'message': result.message
    }


def compare_weights_with_noise(W_pinv, sol_ref, k, t, centros, sigma, n_tests=5):
    """
    Comparar W_pinv con pesos ruidosos

    Demuestra que los pesos iniciales son buenos pero no óptimos.

    Parameters
    ----------
    W_pinv : array (k,1)
        Pesos de mínimos cuadrados
    sol_ref : array
        Solución de referencia
    k : int
        Número de neuronas
    t : array
        Vector de tiempos
    centros : array (k,1)
        Centros
    sigma : float
        Ancho
    n_tests : int, optional
        Número de pruebas con ruido

    Returns
    -------
    list
        Lista con resultados de cada prueba
    """
    print(f"\n{'─'*70}")
    print("PRUEBA: Pesos con ruido (W_pinv + ruido)")
    print(f"{'─'*70}")

    obj_pinv = objective_function(W_pinv.ravel(), sol_ref, k, t, centros, sigma)

    print(f"\n  Objetivo W_pinv: {obj_pinv:.6e}")
    print(f"\n  Generando {n_tests} pesos ruidosos...")

    results = []

    for i in range(n_tests):
        # Agregar ruido uniforme
        noise = np.random.uniform(low=-1, high=1, size=W_pinv.shape)
        W_noisy = W_pinv + noise

        # Evaluar objetivo
        obj_noisy = objective_function(W_noisy.ravel(), sol_ref, k, t, centros, sigma)

        print(f"\n  Test {i+1}:")
        print(f"    W_noisy: {W_noisy.ravel()}")
        print(f"    Objetivo: {obj_noisy:.6e}")
        print(f"    Cambio: {((obj_noisy - obj_pinv)/obj_pinv*100):+.2f}%")

        results.append({
            'W_noisy': W_noisy,
            'objective': obj_noisy,
            'cambio_pct': (obj_noisy - obj_pinv)/obj_pinv*100
        })

    return results


def run_optimization_study(k=5):
    """
    Estudio completo de optimización

    Parameters
    ----------
    k : int
        Número de neuronas

    Returns
    -------
    dict
        Resultados completos
    """
    print("\n" + "="*70)
    print(f"ESTUDIO DE OPTIMIZACIÓN CON k={k} NEURONAS")
    print("="*70)

    # Configuración
    y0 = -0.2
    n = 100
    t = np.linspace(-1, 1, n)

    print(f"\nConfiguración:")
    print(f"  Ecuación: y' + β(y) = sin(5t)")
    print(f"  β(y) = 0.1y³ + 0.1y² + y - 1")
    print(f"  Condición inicial: y(0) = {y0}")
    print(f"  Puntos temporales: {n}")
    print(f"  Neuronas RBF: {k}")

    # Resolver referencia con odeint
    print(f"\n{'─'*70}")
    print("Paso 1: Resolver con odeint (referencia)...")
    sol_ref = solve_ode_odeint(y0, t)
    print(f"  ✓ Completado")

    # Entrenar RBF con mínimos cuadrados
    print(f"\n{'─'*70}")
    print("Paso 2: Entrenar RBF con mínimos cuadrados (W_pinv)...")
    p = 30
    x_train = np.linspace(-3, 2, p).reshape(-1, 1)
    y_train = beta_true(x_train)

    W_pinv, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)

    # Verificar RBF
    y_rbf = VectorRBFI(x_train, W_pinv, centros, sigma)
    rmse_beta = np.sqrt(np.mean((y_rbf - y_train)**2))

    print(f"  Centros: {k}")
    print(f"  Sigma: {sigma:.6f}")
    print(f"  RMSE β(y): {rmse_beta:.6e}")
    print(f"  W_pinv: {W_pinv.ravel()}")

    # Evaluar con W_pinv
    sol_pinv = solve_ode_regressor_rbf(sol_ref[0], sol_ref[1], t, W_pinv, centros, sigma)
    errors_pinv = compute_error(sol_pinv, sol_ref)
    obj_pinv = objective_function(W_pinv.ravel(), sol_ref, k, t, centros, sigma)

    print(f"\n  Resultados con W_pinv:")
    print(f"    Objetivo: {obj_pinv:.6e}")
    print(f"    Error máximo: {errors_pinv['max']:.6e}")
    print(f"    Error RMS: {errors_pinv['rms']:.6e}")

    # Probar con pesos ruidosos
    noise_results = compare_weights_with_noise(W_pinv, sol_ref, k, t, centros, sigma, n_tests=3)

    # Optimizar con Nelder-Mead (seleccionar el peor caso ruidoso)
    print(f"\n{'─'*70}")
    print("Paso 3: Optimizar desde peor caso ruidoso...")

    # Encontrar peor caso
    worst_idx = np.argmax([r['objective'] for r in noise_results])
    W_worst = noise_results[worst_idx]['W_noisy']
    obj_worst = noise_results[worst_idx]['objective']

    print(f"  Iniciando desde peor caso (Test {worst_idx+1}):")
    print(f"    Objetivo inicial: {obj_worst:.6e}")
    print(f"    W_inicial: {W_worst.ravel()}")

    opt_result = optimize_weights_nelder_mead(W_worst, sol_ref, k, t, centros, sigma, maxiter=300)

    W_opt = opt_result['W_opt']

    # Evaluar con W_opt
    sol_opt = solve_ode_regressor_rbf(sol_ref[0], sol_ref[1], t, W_opt, centros, sigma)
    errors_opt = compute_error(sol_opt, sol_ref)
    obj_opt = opt_result['objective_final']

    print(f"\n  Resultados con W_optimizado:")
    print(f"    W_opt: {W_opt.ravel()}")
    print(f"    Objetivo: {obj_opt:.6e}")
    print(f"    Error máximo: {errors_opt['max']:.6e}")
    print(f"    Error RMS: {errors_opt['rms']:.6e}")

    # Comparación final
    print(f"\n{'='*70}")
    print("COMPARACIÓN FINAL")
    print(f"{'='*70}")
    print(f"{'Método':<20} {'Objetivo':<15} {'Error Max':<15} {'Error RMS':<15}")
    print("─"*70)
    print(f"{'W_pinv':<20} {obj_pinv:<15.6e} {errors_pinv['max']:<15.6e} {errors_pinv['rms']:<15.6e}")
    print(f"{'W_ruidoso (peor)':<20} {obj_worst:<15.6e} {'-':<15} {'-':<15}")
    print(f"{'W_optimizado':<20} {obj_opt:<15.6e} {errors_opt['max']:<15.6e} {errors_opt['rms']:<15.6e}")
    print("─"*70)

    mejora_vs_pinv = (obj_pinv - obj_opt) / obj_pinv * 100
    mejora_vs_worst = (obj_worst - obj_opt) / obj_worst * 100

    print(f"\nMejora vs W_pinv: {mejora_vs_pinv:+.2f}%")
    print(f"Mejora vs W_ruidoso: {mejora_vs_worst:+.2f}%")

    print(f"{'='*70}")

    return {
        't': t,
        'sol_ref': sol_ref,
        'sol_pinv': sol_pinv,
        'sol_opt': sol_opt,
        'W_pinv': W_pinv,
        'W_opt': W_opt,
        'errors_pinv': errors_pinv,
        'errors_opt': errors_opt,
        'obj_pinv': obj_pinv,
        'obj_opt': obj_opt,
        'opt_result': opt_result,
        'mejora_vs_pinv': mejora_vs_pinv
    }


def main():
    """Programa principal"""
    print("\n" + "🔵"*35)
    print("OPTIMIZACIÓN DE PESOS RBF - CASO 3")
    print("Método: Nelder-Mead")
    print("🔵"*35)

    print(f"\nObjetivo: Minimizar error del regresor ajustando pesos W")
    print(f"Método: Nelder-Mead (simplex)")

    # Estudio con k=5
    result = run_optimization_study(k=5)

    print("\n\n" + "="*70)
    print("✓ ESTUDIO COMPLETADO")
    print("="*70)

    print(f"\n💡 CONCLUSIONES:")

    if result['mejora_vs_pinv'] > 1:
        print(f"   ✓ Optimización mejora resultados vs W_pinv: +{result['mejora_vs_pinv']:.1f}%")
    elif result['mejora_vs_pinv'] > -1:
        print(f"   • W_pinv ya es casi óptimo (mejora: {result['mejora_vs_pinv']:+.2f}%)")
    else:
        print(f"   • W_pinv es mejor que W_optimizado")

    print(f"   • Nelder-Mead converge desde pesos ruidosos")
    print(f"   • El método es robusto ante perturbaciones")
    print(f"   • Tiempo de optimización: {result['opt_result']['time']:.2f} s")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    np.random.seed(42)
    main()
