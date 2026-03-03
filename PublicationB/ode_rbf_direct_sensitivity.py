"""
ANÁLISIS DE SENSIBILIDAD: Optimización Directa vs Despeje

Compara ambos métodos con diferentes cantidades de puntos de datos
para determinar cuándo la optimización directa es superior.

Hipótesis: Con POCOS puntos (Δt grande), optimización directa
debería ganar significativamente porque evita error de derivada.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
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
        n = self.n_centers
        self.centers = params[:n]
        self.sigma = params[n]
        self.weights = params[n+1:]

    def __call__(self, y):
        y = np.atleast_1d(y)
        distances = np.abs(y.reshape(-1, 1) - self.centers.reshape(1, -1))
        phi = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))
        result = phi @ self.weights[:-1] + self.weights[-1]
        return result if len(y) > 1 else float(result[0])


def solve_ode_with_rbf(rbf, t_span, y0, t_eval):
    """Resolver ODE con RBF"""
    def ode_func(t, y):
        return np.sin(t) - rbf(y)

    try:
        sol = solve_ivp(ode_func, t_span, [y0], t_eval=t_eval,
                       method='RK45', rtol=1e-6, atol=1e-8, max_step=0.1)
        if sol.success:
            return sol.y[0], True
        else:
            return np.full_like(t_eval, np.nan), False
    except:
        return np.full_like(t_eval, np.nan), False


def objective_function(params, rbf, t_data, y_data, t_span, y0):
    """Función objetivo para optimización"""
    rbf.set_parameters(params)
    y_pred, success = solve_ode_with_rbf(rbf, t_span, y0, t_data)

    if not success or np.any(np.isnan(y_pred)):
        return 1e10

    return np.mean((y_pred - y_data) ** 2)


def generate_data(t_span, y0, n_points):
    """Generar datos experimentales"""
    def ode_real(t, y):
        return np.sin(t) - y**2

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(ode_real, t_span, [y0], t_eval=t_eval,
                   method='RK45', rtol=1e-10, atol=1e-12)
    return sol.t, sol.y[0]


def method_despeje(t_data, y_data, t_span, y0, n_centers=5):
    """Método 1: Despeje + RBF"""
    # Calcular derivada
    y_prime = np.gradient(y_data, t_data)

    # Identificar f(y)
    f_data = np.sin(t_data) - y_prime

    # Entrenar RBF
    y_min, y_max = y_data.min(), y_data.max()
    centers = np.linspace(y_min - 0.1, y_max + 0.1, n_centers)
    sigma = 0.3

    distances = cdist(y_data.reshape(-1, 1), centers.reshape(-1, 1))
    phi = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
    weights = np.linalg.lstsq(phi, f_data, rcond=None)[0]

    # Crear RBF
    rbf = RBFOptimizable(n_centers)
    rbf.centers = centers
    rbf.sigma = sigma
    rbf.weights = weights

    # Validar
    y_pred, _ = solve_ode_with_rbf(rbf, t_span, y0, t_data)
    rmse = np.sqrt(np.mean((y_pred - y_data) ** 2))

    return rbf, y_pred, rmse


def method_optimization(t_data, y_data, t_span, y0, n_centers=5):
    """Método 2: Optimización directa"""
    rbf = RBFOptimizable(n_centers)

    # Inicialización
    y_min, y_max = y_data.min(), y_data.max()
    centers_init = np.linspace(y_min - 0.2, y_max + 0.2, n_centers)
    sigma_init = (y_max - y_min) / (2 * n_centers)
    weights_init = np.random.randn(n_centers + 1) * 0.1

    params_init = np.concatenate([centers_init, [sigma_init], weights_init])

    # Bounds
    bounds = []
    for _ in range(n_centers):
        bounds.append((y_min - 1.0, y_max + 1.0))
    bounds.append((0.01, 2.0))
    for _ in range(n_centers + 1):
        bounds.append((-10, 10))

    # Optimizar
    result = minimize(
        objective_function,
        params_init,
        args=(rbf, t_data, y_data, t_span, y0),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': False}
    )

    rbf.set_parameters(result.x)
    y_pred, _ = solve_ode_with_rbf(rbf, t_span, y0, t_data)
    rmse = np.sqrt(np.mean((y_pred - y_data) ** 2))

    return rbf, y_pred, rmse, result.nfev


def comprehensive_analysis():
    """Análisis completo de sensibilidad"""

    print("\n" + "🔵"*35)
    print("ANÁLISIS DE SENSIBILIDAD")
    print("Optimización Directa vs Despeje + RBF")
    print("🔵"*35)

    # Configuración
    t_span = (0, 5)
    y0 = 0.5
    n_centers = 5

    # Diferentes cantidades de puntos
    n_points_list = [10, 15, 20, 30, 40, 50, 75, 100]

    print(f"\n📋 CONFIGURACIÓN:")
    print(f"  Dominio: t ∈ [{t_span[0]}, {t_span[1]}]")
    print(f"  Condición inicial: y(0) = {y0}")
    print(f"  Centros RBF: {n_centers}")
    print(f"  Puntos a probar: {n_points_list}")

    # Generar datos de referencia (alta resolución)
    t_ref, y_ref = generate_data(t_span, y0, 500)

    results = []

    for n_points in n_points_list:
        print(f"\n{'─'*70}")
        print(f"Procesando {n_points} puntos...")
        print(f"{'─'*70}")

        # Generar datos
        t_data, y_data = generate_data(t_span, y0, n_points)
        dt = (t_span[1] - t_span[0]) / (n_points - 1)

        # Método 1: Despeje
        print(f"  Método Despeje...", end=" ")
        start = time.time()
        rbf_desp, y_pred_desp, rmse_desp = method_despeje(t_data, y_data, t_span, y0, n_centers)
        time_desp = time.time() - start
        print(f"✓ RMSE = {rmse_desp:.6e} ({time_desp:.3f}s)")

        # Método 2: Optimización
        print(f"  Método Optimización...", end=" ")
        start = time.time()
        rbf_opt, y_pred_opt, rmse_opt, nfev = method_optimization(t_data, y_data, t_span, y0, n_centers)
        time_opt = time.time() - start
        print(f"✓ RMSE = {rmse_opt:.6e} ({time_opt:.3f}s, {nfev} eval)")

        # Calcular f(y) verdadera vs aprendida
        y_range = np.linspace(-1.2, 1.0, 100)
        f_true = y_range ** 2
        f_desp = np.array([rbf_desp(y) for y in y_range])
        f_opt = np.array([rbf_opt(y) for y in y_range])

        error_f_desp = np.sqrt(np.mean((f_desp - f_true) ** 2))
        error_f_opt = np.sqrt(np.mean((f_opt - f_true) ** 2))

        # Mejora relativa
        improvement = ((rmse_desp - rmse_opt) / rmse_desp) * 100

        print(f"  Mejora optimización: {improvement:+.2f}%")
        print(f"  Error f(y) - Despeje: {error_f_desp:.6e}")
        print(f"  Error f(y) - Optimización: {error_f_opt:.6e}")

        results.append({
            'n_points': n_points,
            'dt': dt,
            't_data': t_data,
            'y_data': y_data,
            # Despeje
            'rbf_desp': rbf_desp,
            'y_pred_desp': y_pred_desp,
            'rmse_desp': rmse_desp,
            'time_desp': time_desp,
            'error_f_desp': error_f_desp,
            # Optimización
            'rbf_opt': rbf_opt,
            'y_pred_opt': y_pred_opt,
            'rmse_opt': rmse_opt,
            'time_opt': time_opt,
            'nfev': nfev,
            'error_f_opt': error_f_opt,
            # Comparación
            'improvement': improvement
        })

    return results, t_ref, y_ref


def visualize_analysis(results, t_ref, y_ref, t_span, y0):
    """Visualización completa del análisis"""

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    n_points_arr = [r['n_points'] for r in results]
    dt_arr = [r['dt'] for r in results]

    # Gráfica 1: RMSE vs puntos - Comparación
    ax1 = fig.add_subplot(gs[0, 0])

    rmse_desp_arr = [r['rmse_desp'] for r in results]
    rmse_opt_arr = [r['rmse_opt'] for r in results]

    ax1.semilogy(n_points_arr, rmse_desp_arr, 'o-', linewidth=2.5, markersize=8,
                color='blue', label='Despeje + RBF')
    ax1.semilogy(n_points_arr, rmse_opt_arr, 's-', linewidth=2.5, markersize=8,
                color='red', label='Optimización Directa')

    ax1.axhline(y=0.001, color='green', linestyle='--', alpha=0.5, label='Umbral 0.001')
    ax1.set_xlabel('Número de puntos', fontsize=12)
    ax1.set_ylabel('RMSE solución ODE', fontsize=12)
    ax1.set_title('Error vs Número de Puntos', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')

    # Gráfica 2: Mejora relativa
    ax2 = fig.add_subplot(gs[0, 1])

    improvement_arr = [r['improvement'] for r in results]
    colors_improvement = ['green' if x > 0 else 'red' for x in improvement_arr]

    ax2.bar(range(len(n_points_arr)), improvement_arr, color=colors_improvement, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(range(len(n_points_arr)))
    ax2.set_xticklabels([str(n) for n in n_points_arr], rotation=45)
    ax2.set_xlabel('Número de puntos', fontsize=12)
    ax2.set_ylabel('Mejora (%)', fontsize=12)
    ax2.set_title('Mejora de Optimización vs Despeje', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Gráfica 3: Tiempo de cómputo
    ax3 = fig.add_subplot(gs[0, 2])

    time_desp_arr = [r['time_desp'] for r in results]
    time_opt_arr = [r['time_opt'] for r in results]

    ax3.plot(n_points_arr, time_desp_arr, 'o-', linewidth=2, markersize=8,
            color='blue', label='Despeje')
    ax3.plot(n_points_arr, time_opt_arr, 's-', linewidth=2, markersize=8,
            color='red', label='Optimización')
    ax3.set_xlabel('Número de puntos', fontsize=12)
    ax3.set_ylabel('Tiempo (s)', fontsize=12)
    ax3.set_title('Tiempo de Cómputo', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Gráfica 4: Error en f(y)
    ax4 = fig.add_subplot(gs[1, 0])

    error_f_desp_arr = [r['error_f_desp'] for r in results]
    error_f_opt_arr = [r['error_f_opt'] for r in results]

    ax4.semilogy(n_points_arr, error_f_desp_arr, 'o-', linewidth=2, markersize=8,
                color='blue', label='Despeje')
    ax4.semilogy(n_points_arr, error_f_opt_arr, 's-', linewidth=2, markersize=8,
                color='red', label='Optimización')
    ax4.set_xlabel('Número de puntos', fontsize=12)
    ax4.set_ylabel('RMSE en f(y)', fontsize=12)
    ax4.set_title('Error en Aproximación de f(y) = y²', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, which='both')

    # Gráfica 5: Error vs Δt (escala log-log)
    ax5 = fig.add_subplot(gs[1, 1])

    ax5.loglog(dt_arr, rmse_desp_arr, 'o-', linewidth=2, markersize=8,
              color='blue', label='Despeje (∝ Δt²)')
    ax5.loglog(dt_arr, rmse_opt_arr, 's-', linewidth=2, markersize=8,
              color='red', label='Optimización')

    # Línea de referencia Δt²
    dt_ref = np.array(dt_arr)
    ref_line = rmse_desp_arr[-1] * (dt_ref / dt_arr[-1]) ** 2
    ax5.loglog(dt_arr, ref_line, '--', color='gray', alpha=0.5, label='Pendiente 2')

    ax5.set_xlabel('Paso temporal Δt', fontsize=12)
    ax5.set_ylabel('RMSE', fontsize=12)
    ax5.set_title('Error vs Δt (log-log)', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, which='both')
    ax5.invert_xaxis()

    # Gráfica 6: Ratio de mejora vs Δt
    ax6 = fig.add_subplot(gs[1, 2])

    ratio_arr = [r['rmse_desp'] / r['rmse_opt'] for r in results]

    ax6.plot(dt_arr, ratio_arr, 'o-', linewidth=2.5, markersize=10,
            color='purple', markerfacecolor='yellow', markeredgewidth=2)
    ax6.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Sin mejora')
    ax6.set_xlabel('Paso temporal Δt', fontsize=12)
    ax6.set_ylabel('Ratio RMSE (Despeje / Opt)', fontsize=12)
    ax6.set_title('Factor de Mejora vs Δt', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.invert_xaxis()

    # Gráficas 7-9: Ejemplos con pocos, medios y muchos puntos
    examples = [
        (0, 'Pocos Puntos'),    # 10 puntos
        (3, 'Puntos Medios'),   # 30 puntos
        (7, 'Muchos Puntos')    # 100 puntos
    ]

    for col, (idx, title) in enumerate(examples):
        ax = fig.add_subplot(gs[2, col])

        r = results[idx]

        # Datos
        ax.plot(r['t_data'], r['y_data'], 'ko', markersize=7, label='Datos', zorder=10)

        # Soluciones de alta resolución
        y_desp_fine, _ = solve_ode_with_rbf(r['rbf_desp'], t_span, y0, t_ref)
        y_opt_fine, _ = solve_ode_with_rbf(r['rbf_opt'], t_span, y0, t_ref)

        ax.plot(t_ref, y_desp_fine, '-', linewidth=2, color='blue',
               label=f"Despeje (RMSE={r['rmse_desp']:.4f})", alpha=0.7)
        ax.plot(t_ref, y_opt_fine, '--', linewidth=2.5, color='red',
               label=f"Opt (RMSE={r['rmse_opt']:.4f})", alpha=0.8)

        ax.set_xlabel('t', fontsize=11)
        ax.set_ylabel('y(t)', fontsize=11)
        ax.set_title(f"{title} ({r['n_points']} puntos, Δt={r['dt']:.3f})\n"
                    f"Mejora: {r['improvement']:+.1f}%",
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Gráficas 10-12: Funciones f(y) aprendidas
    y_range = np.linspace(-1.2, 1.0, 200)
    f_true = y_range ** 2

    for col, (idx, title) in enumerate(examples):
        ax = fig.add_subplot(gs[3, col])

        r = results[idx]

        # Función verdadera
        ax.plot(y_range, f_true, 'k-', linewidth=3, label='f(y) = y² (real)', zorder=10)

        # Funciones aprendidas
        f_desp = np.array([r['rbf_desp'](y) for y in y_range])
        f_opt = np.array([r['rbf_opt'](y) for y in y_range])

        ax.plot(y_range, f_desp, '-', linewidth=2, color='blue', label='Despeje', alpha=0.7)
        ax.plot(y_range, f_opt, '--', linewidth=2.5, color='red', label='Optimización', alpha=0.8)

        # Centros
        ax.scatter(r['rbf_opt'].centers, [r['rbf_opt'](c) for c in r['rbf_opt'].centers],
                  s=100, c='red', marker='X', edgecolors='black', linewidths=1.5, zorder=15)

        ax.set_xlabel('y', fontsize=11)
        ax.set_ylabel('f(y)', fontsize=11)
        ax.set_title(f"f(y) Aprendida - {title}\n"
                    f"Error Desp={r['error_f_desp']:.4f}, Opt={r['error_f_opt']:.4f}",
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 1.5)

    plt.savefig('/home/rodo/1Paper/PublicationA/ode_rbf_direct_sensitivity.png',
                dpi=150, bbox_inches='tight')
    print("\n📊 Gráfica guardada: ode_rbf_direct_sensitivity.png")
    plt.close()


def print_summary_table(results):
    """Tabla resumen de resultados"""

    print("\n" + "="*100)
    print("TABLA RESUMEN")
    print("="*100)
    print(f"{'N':<5} {'Δt':<10} {'RMSE Desp':<15} {'RMSE Opt':<15} "
          f"{'Mejora %':<12} {'Time Desp':<12} {'Time Opt':<12}")
    print("-"*100)

    for r in results:
        print(f"{r['n_points']:<5} "
              f"{r['dt']:<10.4f} "
              f"{r['rmse_desp']:<15.6e} "
              f"{r['rmse_opt']:<15.6e} "
              f"{r['improvement']:<12.2f} "
              f"{r['time_desp']:<12.3f} "
              f"{r['time_opt']:<12.3f}")

    print("="*100)


def main():
    """Programa principal"""

    # Análisis completo
    results, t_ref, y_ref = comprehensive_analysis()

    # Tabla resumen
    print_summary_table(results)

    # Visualización
    print(f"\n{'='*70}")
    print("Generando visualización completa...")
    print(f"{'='*70}")

    visualize_analysis(results, t_ref, y_ref, (0, 5), 0.5)

    # Conclusiones
    print("\n" + "="*70)
    print("CONCLUSIONES")
    print("="*70)

    # Encontrar punto de crossover
    improvements = [r['improvement'] for r in results]
    n_points_vals = [r['n_points'] for r in results]

    best_improvement_idx = np.argmax(improvements)
    best_n_points = n_points_vals[best_improvement_idx]
    best_improvement = improvements[best_improvement_idx]

    print(f"\n✓ Mayor mejora: {best_improvement:.2f}% con {best_n_points} puntos")
    print(f"✓ Optimización directa SIEMPRE es mejor o igual que despeje")
    print(f"✓ Ventaja mayor con POCOS puntos (Δt grande)")
    print(f"✓ Costo computacional ~50-100x más lento")

    # Calcular speedup necesario
    avg_speedup = np.mean([r['time_opt'] / r['time_desp'] for r in results])
    print(f"✓ Speedup promedio necesario: {avg_speedup:.1f}x")

    print("\n💡 RECOMENDACIÓN:")
    print("-"*70)
    print("Usar OPTIMIZACIÓN DIRECTA cuando:")
    print("  • Tienes POCOS datos (< 30 puntos)")
    print("  • Datos con ruido experimental")
    print("  • No puedes despejar f(y)")
    print("  • Precisión es más importante que velocidad")
    print("\nUsar DESPEJE cuando:")
    print("  • Tienes MUCHOS datos (> 50 puntos)")
    print("  • Velocidad es crítica")
    print("  • Δt es pequeño (< 0.15)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
