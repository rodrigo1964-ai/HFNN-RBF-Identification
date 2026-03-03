"""
AJUSTE DIRECTO DE RBF DENTRO DE LA ODE

Método alternativo: En lugar de despejar f(y) de la ecuación,
optimizamos los parámetros de la RBF directamente minimizando
el residuo entre la solución de la ODE y los datos observados.

Ventajas:
- No requiere calcular derivadas de los datos
- Más robusto al ruido experimental
- Funciona cuando f(y) no se puede despejar

Problema de optimización:
  min_{w, c, σ} ||y_ode(t; RBF(w,c,σ)) - y_data||²
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import time


class RBFOptimizable:
    """RBF con parámetros optimizables"""

    def __init__(self, n_centers):
        self.n_centers = n_centers
        self.centers = None
        self.sigma = None
        self.weights = None

    def set_parameters(self, params):
        """
        Establecer parámetros desde un vector
        params = [c1, c2, ..., cM, sigma, w1, w2, ..., wM, w0]
        """
        n = self.n_centers
        self.centers = params[:n]
        self.sigma = params[n]
        self.weights = params[n+1:]

    def __call__(self, y):
        """Evaluar RBF(y)"""
        y = np.atleast_1d(y)
        distances = np.abs(y.reshape(-1, 1) - self.centers.reshape(1, -1))
        phi = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))
        result = phi @ self.weights[:-1] + self.weights[-1]
        return result if len(y) > 1 else float(result[0])


def solve_ode_with_rbf(rbf, t_span, y0, t_eval):
    """
    Resolver ODE: y' + RBF(y) = sin(t)
    con la RBF dada
    """
    def ode_func(t, y):
        return np.sin(t) - rbf(y)

    try:
        sol = solve_ivp(
            ode_func,
            t_span,
            [y0],
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-8,
            max_step=0.1
        )

        if sol.success:
            return sol.y[0], True
        else:
            return np.full_like(t_eval, np.nan), False

    except Exception:
        return np.full_like(t_eval, np.nan), False


def objective_function(params, rbf, t_data, y_data, t_span, y0, reg_strength=0.0):
    """
    Función objetivo: error entre solución ODE y datos

    Args:
        params: parámetros de la RBF [centros, sigma, pesos]
        reg_strength: regularización L2 en los pesos
    """
    # Establecer parámetros en la RBF
    rbf.set_parameters(params)

    # Resolver ODE con estos parámetros
    y_pred, success = solve_ode_with_rbf(rbf, t_span, y0, t_data)

    if not success or np.any(np.isnan(y_pred)):
        return 1e10  # Penalización por fallo

    # Error cuadrático medio
    mse = np.mean((y_pred - y_data) ** 2)

    # Regularización (opcional)
    if reg_strength > 0:
        weights = params[rbf.n_centers + 1:]
        reg_term = reg_strength * np.sum(weights[:-1] ** 2)  # No regularizar bias
        return mse + reg_term

    return mse


def optimize_rbf_direct(t_data, y_data, t_span, y0, n_centers=5,
                        method='L-BFGS-B', max_iter=100):
    """
    Método 1: Optimización con gradiente (L-BFGS-B)

    Optimiza parámetros de RBF minimizando error en ODE
    """
    print(f"\n{'='*70}")
    print(f"MÉTODO 1: Optimización Directa con {method}")
    print(f"{'='*70}")
    print(f"  Centros RBF: {n_centers}")
    print(f"  Método: {method}")
    print(f"  Datos: {len(t_data)} puntos")

    # Inicialización
    rbf = RBFOptimizable(n_centers)

    # Parámetros iniciales
    y_min, y_max = y_data.min(), y_data.max()
    centers_init = np.linspace(y_min - 0.2, y_max + 0.2, n_centers)
    sigma_init = (y_max - y_min) / (2 * n_centers)
    weights_init = np.random.randn(n_centers + 1) * 0.1

    params_init = np.concatenate([centers_init, [sigma_init], weights_init])

    print(f"  Parámetros totales: {len(params_init)}")
    print(f"    - Centros: {n_centers}")
    print(f"    - Sigma: 1")
    print(f"    - Pesos: {n_centers + 1}")

    # Bounds
    bounds = []
    # Centros: pueden moverse en un rango amplio
    for _ in range(n_centers):
        bounds.append((y_min - 1.0, y_max + 1.0))
    # Sigma: positivo, rango razonable
    bounds.append((0.01, 2.0))
    # Pesos: sin restricción (pero razonables)
    for _ in range(n_centers + 1):
        bounds.append((-10, 10))

    # Error inicial
    error_init = objective_function(params_init, rbf, t_data, y_data, t_span, y0)
    print(f"  Error inicial: {error_init:.6e}")

    print(f"\n  Optimizando...")
    start_time = time.time()

    # Optimización
    result = minimize(
        objective_function,
        params_init,
        args=(rbf, t_data, y_data, t_span, y0, 0.0),
        method=method,
        bounds=bounds,
        options={'maxiter': max_iter, 'disp': False}
    )

    elapsed = time.time() - start_time

    # Establecer parámetros óptimos
    rbf.set_parameters(result.x)

    # Evaluar solución final
    y_pred, success = solve_ode_with_rbf(rbf, t_span, y0, t_data)

    print(f"\n  Resultados:")
    print(f"    Estado: {result.message}")
    print(f"    Éxito: {result.success}")
    print(f"    Iteraciones: {result.nit}")
    print(f"    Evaluaciones: {result.nfev}")
    print(f"    Tiempo: {elapsed:.2f} s")
    print(f"    Error final: {result.fun:.6e}")

    if success:
        rmse = np.sqrt(np.mean((y_pred - y_data) ** 2))
        max_error = np.max(np.abs(y_pred - y_data))
        print(f"    RMSE: {rmse:.6e}")
        print(f"    Error máximo: {max_error:.6e}")

    return rbf, result, y_pred


def optimize_rbf_global(t_data, y_data, t_span, y0, n_centers=5, max_iter=50):
    """
    Método 2: Optimización global (Differential Evolution)

    Más robusto a mínimos locales, pero más lento
    """
    print(f"\n{'='*70}")
    print(f"MÉTODO 2: Optimización Global (Differential Evolution)")
    print(f"{'='*70}")
    print(f"  Centros RBF: {n_centers}")
    print(f"  Datos: {len(t_data)} puntos")
    print(f"  Máx iteraciones: {max_iter}")

    # Inicialización
    rbf = RBFOptimizable(n_centers)

    y_min, y_max = y_data.min(), y_data.max()

    # Bounds
    bounds = []
    # Centros
    for _ in range(n_centers):
        bounds.append((y_min - 1.0, y_max + 1.0))
    # Sigma
    bounds.append((0.05, 1.5))
    # Pesos
    for _ in range(n_centers + 1):
        bounds.append((-5, 5))

    print(f"  Parámetros totales: {len(bounds)}")
    print(f"\n  Optimizando (esto puede tardar)...")
    start_time = time.time()

    # Differential Evolution
    result = differential_evolution(
        objective_function,
        bounds,
        args=(rbf, t_data, y_data, t_span, y0, 0.0),
        strategy='best1bin',
        maxiter=max_iter,
        popsize=15,
        tol=1e-7,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False,
        workers=1
    )

    elapsed = time.time() - start_time

    # Establecer parámetros óptimos
    rbf.set_parameters(result.x)
    y_pred, success = solve_ode_with_rbf(rbf, t_span, y0, t_data)

    print(f"\n  Resultados:")
    print(f"    Estado: {result.message}")
    print(f"    Éxito: {result.success}")
    print(f"    Iteraciones: {result.nit}")
    print(f"    Evaluaciones: {result.nfev}")
    print(f"    Tiempo: {elapsed:.2f} s")
    print(f"    Error final: {result.fun:.6e}")

    if success:
        rmse = np.sqrt(np.mean((y_pred - y_data) ** 2))
        max_error = np.max(np.abs(y_pred - y_data))
        print(f"    RMSE: {rmse:.6e}")
        print(f"    Error máximo: {max_error:.6e}")

    return rbf, result, y_pred


def generate_experimental_data(t_span, y0, n_points, noise=0.0):
    """Generar datos experimentales"""
    def ode_real(t, y):
        return np.sin(t) - y**2

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(ode_real, t_span, [y0], t_eval=t_eval, method='RK45',
                    rtol=1e-10, atol=1e-12)

    t_data = sol.t
    y_data = sol.y[0]

    if noise > 0:
        y_data += np.random.normal(0, noise, len(y_data))

    return t_data, y_data


def compare_methods(t_data, y_data, t_span, y0):
    """Comparar todos los métodos"""

    # Método 1: Despeje + RBF (método anterior)
    print(f"\n{'🔵'*35}")
    print("MÉTODO DE REFERENCIA: Despeje + RBF")
    print(f"{'🔵'*35}")

    # Calcular derivada
    y_prime = np.gradient(y_data, t_data)
    # Identificar f(y)
    f_data = np.sin(t_data) - y_prime

    # Entrenar RBF estándar
    from scipy.spatial.distance import cdist

    n_centers = 8
    y_min, y_max = y_data.min(), y_data.max()
    centers_ref = np.linspace(y_min, y_max, n_centers)
    sigma_ref = 0.3

    distances = cdist(y_data.reshape(-1, 1), centers_ref.reshape(-1, 1))
    phi = np.exp(-(distances ** 2) / (2 * sigma_ref ** 2))
    phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
    weights_ref = np.linalg.lstsq(phi, f_data, rcond=None)[0]

    # Crear RBF de referencia
    rbf_ref = RBFOptimizable(n_centers)
    rbf_ref.centers = centers_ref
    rbf_ref.sigma = sigma_ref
    rbf_ref.weights = weights_ref

    y_pred_ref, _ = solve_ode_with_rbf(rbf_ref, t_span, y0, t_data)
    rmse_ref = np.sqrt(np.mean((y_pred_ref - y_data) ** 2))
    print(f"  RMSE: {rmse_ref:.6e}")

    # Método 2: Optimización directa con gradiente
    rbf_opt, result_opt, y_pred_opt = optimize_rbf_direct(
        t_data, y_data, t_span, y0, n_centers=5, method='L-BFGS-B', max_iter=100
    )

    # Método 3: Optimización global
    rbf_global, result_global, y_pred_global = optimize_rbf_global(
        t_data, y_data, t_span, y0, n_centers=5, max_iter=30
    )

    return {
        'reference': {'rbf': rbf_ref, 'y_pred': y_pred_ref, 'rmse': rmse_ref},
        'gradient': {'rbf': rbf_opt, 'y_pred': y_pred_opt, 'result': result_opt},
        'global': {'rbf': rbf_global, 'y_pred': y_pred_global, 'result': result_global}
    }


def visualize_comparison(t_data, y_data, results, t_span, y0):
    """Visualizar comparación de métodos"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Solución de referencia de alta resolución
    t_fine = np.linspace(t_span[0], t_span[1], 500)

    # Gráficas superiores: Soluciones
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[0, idx]

        # Datos experimentales
        ax.plot(t_data, y_data, 'ko', markersize=6, label='Datos', zorder=10)

        # Predicción con alta resolución
        y_pred_fine, _ = solve_ode_with_rbf(data['rbf'], t_span, y0, t_fine)
        ax.plot(t_fine, y_pred_fine, 'r-', linewidth=2.5, label='RBF optimizada', alpha=0.8)

        # Título con RMSE
        if 'rmse' in data:
            rmse = data['rmse']
        else:
            rmse = np.sqrt(np.mean((data['y_pred'] - y_data) ** 2))

        titles = {
            'reference': 'Despeje + RBF\n(Método Estándar)',
            'gradient': 'Optimización Directa\n(L-BFGS-B)',
            'global': 'Optimización Global\n(Diff. Evolution)'
        }

        ax.set_title(f"{titles[name]}\nRMSE = {rmse:.6e}", fontsize=11, fontweight='bold')
        ax.set_xlabel('t', fontsize=11)
        ax.set_ylabel('y(t)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Gráficas inferiores: Función f(y) aprendida
    y_range = np.linspace(y_data.min() - 0.2, y_data.max() + 0.2, 200)
    f_true = y_range ** 2

    for idx, (name, data) in enumerate(results.items()):
        ax = axes[1, idx]

        # Función verdadera
        ax.plot(y_range, f_true, 'k-', linewidth=3, label='f(y) = y² (real)', zorder=10)

        # Función aprendida
        f_learned = np.array([data['rbf'](y_val) for y_val in y_range])
        ax.plot(y_range, f_learned, 'r--', linewidth=2.5, label='RBF(y)', alpha=0.8)

        # Centros
        rbf = data['rbf']
        ax.scatter(rbf.centers, [rbf(c) for c in rbf.centers],
                  s=150, c='red', marker='X', edgecolors='black',
                  linewidths=2, zorder=15, label='Centros')

        # Error de aproximación
        error_f = np.sqrt(np.mean((f_learned - f_true) ** 2))

        ax.set_title(f"Función Aprendida\nRMSE f(y) = {error_f:.6e}",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('y', fontsize=11)
        ax.set_ylabel('f(y)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 1.5)

    plt.tight_layout()
    plt.savefig('/home/rodo/1Paper/PublicationA/ode_rbf_direct_optimization.png',
                dpi=150, bbox_inches='tight')
    print("\n📊 Gráfica guardada: ode_rbf_direct_optimization.png")
    plt.close()


def main():
    """Programa principal"""

    print("\n" + "🟢"*35)
    print("AJUSTE DIRECTO DE RBF DENTRO DE LA ODE")
    print("Sin calcular derivadas de los datos")
    print("🟢"*35)

    # Configuración
    t_span = (0, 5)
    y0 = 0.5
    n_points = 40
    noise_level = 0.0

    print(f"\n📋 CONFIGURACIÓN:")
    print(f"  Dominio: t ∈ [{t_span[0]}, {t_span[1]}]")
    print(f"  Condición inicial: y(0) = {y0}")
    print(f"  Puntos de datos: {n_points}")
    print(f"  Ruido: {noise_level}")

    # Generar datos
    print(f"\n{'─'*70}")
    print("Generando datos experimentales...")
    t_data, y_data = generate_experimental_data(t_span, y0, n_points, noise_level)
    print(f"  ✓ {len(t_data)} puntos generados")
    print(f"  Rango de y: [{y_data.min():.4f}, {y_data.max():.4f}]")

    # Comparar métodos
    results = compare_methods(t_data, y_data, t_span, y0)

    # Visualizar
    print(f"\n{'='*70}")
    print("Generando visualización comparativa...")
    print(f"{'='*70}")
    visualize_comparison(t_data, y_data, results, t_span, y0)

    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*70}")
    print(f"{'Método':<35} {'RMSE':>15} {'Tiempo':>15}")
    print("─"*70)

    for name, data in results.items():
        if 'rmse' in data:
            rmse = data['rmse']
            tiempo = "< 0.1 s"
        else:
            rmse = np.sqrt(np.mean((data['y_pred'] - y_data) ** 2))
            if 'result' in data and hasattr(data['result'], 'nfev'):
                tiempo = f"{data['result'].nfev} eval"

        names_es = {
            'reference': 'Despeje + RBF',
            'gradient': 'Optimización Directa (L-BFGS-B)',
            'global': 'Optimización Global (Diff. Evol.)'
        }
        print(f"{names_es[name]:<35} {rmse:>15.6e} {tiempo:>15}")

    print("="*70)

    print("\n💡 CONCLUSIONES:")
    print("─"*70)
    print("✓ Optimización directa EVITA calcular derivadas de los datos")
    print("✓ Más robusto al ruido experimental")
    print("✓ Funciona cuando f(y) no se puede despejar")
    print("✓ Requiere más tiempo de cómputo (múltiples integraciones ODE)")
    print("✓ Optimización global es más robusta pero más lenta")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
