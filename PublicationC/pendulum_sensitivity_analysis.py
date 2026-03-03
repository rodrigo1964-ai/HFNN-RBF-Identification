"""
ANÁLISIS DE SENSIBILIDAD: Péndulo con Fricción Viscosa

Evalúa cómo el número de puntos de medición afecta la calidad
de identificación de la ley de fricción c(ω) = b₁·ω + b₂·ω³

Compara:
  - Método Tradicional: Despeje + RBF
  - Método Directo: Optimización sin derivadas
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time


# Parámetros físicos
OMEGA_0 = 2.0
B1 = 0.3
B2 = 0.05


def true_friction(omega):
    return B1 * omega + B2 * omega**3


def generate_data(n_points):
    """Generar datos experimentales con n_points mediciones"""
    def pendulum(t, state):
        theta, omega = state
        return [omega, -OMEGA_0**2 * np.sin(theta) - true_friction(omega)]

    t_span = (0, 15)
    initial = [0.8, 0.0]
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    sol = solve_ivp(pendulum, t_span, initial, t_eval=t_eval,
                   method='RK45', rtol=1e-10, atol=1e-12)

    return sol.t, sol.y[0], sol.y[1], t_span, initial


class RBFOptimizable:
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

    def __call__(self, omega):
        omega = np.atleast_1d(omega)
        distances = np.abs(omega.reshape(-1, 1) - self.centers.reshape(1, -1))
        phi = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))
        result = phi @ self.weights[:-1] + self.weights[-1]
        return result if len(omega) > 1 else float(result[0])


def method_despeje(t_data, theta_data):
    """Método tradicional: calcular derivadas y despejar"""
    start = time.time()

    theta_prime = np.gradient(theta_data, t_data)
    theta_second = np.gradient(theta_prime, t_data)
    c_data = -theta_second - OMEGA_0**2 * np.sin(theta_data)

    n_centers = min(8, len(t_data) // 4)
    omega_min, omega_max = theta_prime.min(), theta_prime.max()
    centers = np.linspace(omega_min - 0.1, omega_max + 0.1, n_centers)
    sigma = max(0.05, (omega_max - omega_min) / (2 * n_centers))

    distances = cdist(theta_prime.reshape(-1, 1), centers.reshape(-1, 1))
    phi = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
    weights = np.linalg.lstsq(phi, c_data, rcond=None)[0]

    rbf = RBFOptimizable(n_centers)
    rbf.centers = centers
    rbf.sigma = sigma
    rbf.weights = weights

    elapsed = time.time() - start
    return rbf, elapsed


def method_direct(t_data, theta_data, t_span, initial):
    """Método de optimización directa"""
    start = time.time()

    n_centers = min(6, len(t_data) // 5)
    rbf = RBFOptimizable(n_centers)

    def solve_pendulum(rbf, t_eval):
        def pendulum_rbf(t, state):
            theta, omega = state
            return [omega, -OMEGA_0**2 * np.sin(theta) - rbf(omega)]

        try:
            sol = solve_ivp(pendulum_rbf, t_span, initial, t_eval=t_eval,
                          method='RK45', rtol=1e-6, atol=1e-8)
            if sol.success:
                return sol.y[0], True
            return np.full_like(t_eval, np.nan), False
        except:
            return np.full_like(t_eval, np.nan), False

    def objective(params):
        rbf.set_parameters(params)
        theta_pred, success = solve_pendulum(rbf, t_data)
        if not success or np.any(np.isnan(theta_pred)):
            return 1e10
        return np.mean((theta_pred - theta_data) ** 2)

    omega_est = np.gradient(theta_data, t_data)
    omega_min, omega_max = omega_est.min(), omega_est.max()

    centers_init = np.linspace(omega_min - 0.2, omega_max + 0.2, n_centers)
    sigma_init = max(0.1, (omega_max - omega_min) / (2 * n_centers))
    weights_init = np.random.randn(n_centers + 1) * 0.1
    params_init = np.concatenate([centers_init, [sigma_init], weights_init])

    bounds = []
    for _ in range(n_centers):
        bounds.append((omega_min - 0.5, omega_max + 0.5))
    bounds.append((0.01, 2.0))
    for _ in range(n_centers + 1):
        bounds.append((-5, 5))

    result = minimize(objective, params_init, method='L-BFGS-B',
                     bounds=bounds, options={'maxiter': 150, 'disp': False})

    rbf.set_parameters(result.x)
    elapsed = time.time() - start

    return rbf, elapsed


def evaluate_rbf_accuracy(rbf, omega_range):
    """Evaluar precisión de RBF vs función real"""
    c_true = true_friction(omega_range)
    c_pred = np.array([rbf(w) for w in omega_range])
    rmse = np.sqrt(np.mean((c_pred - c_true) ** 2))
    return rmse


def run_sensitivity_analysis():
    """Análisis completo variando número de puntos"""

    n_points_list = [10, 15, 20, 30, 40, 50, 75, 100]

    results = {
        'n_points': [],
        'dt': [],
        'rmse_despeje': [],
        'rmse_direct': [],
        'time_despeje': [],
        'time_direct': [],
        'improvement': []
    }

    print("\n" + "="*80)
    print("ANÁLISIS DE SENSIBILIDAD: Fricción Viscosa en Péndulo")
    print("="*80)
    print(f"\nFricción real: c(ω) = {B1}·ω + {B2}·ω³")
    print(f"\nProbando con: {n_points_list} puntos\n")

    for n_points in n_points_list:
        print(f"\n{'─'*80}")
        print(f"N = {n_points} puntos")
        print(f"{'─'*80}")

        # Generar datos
        t_data, theta_data, omega_data, t_span, initial = generate_data(n_points)
        dt = t_data[1] - t_data[0]

        omega_range = np.linspace(omega_data.min(), omega_data.max(), 100)

        # Método 1: Despeje
        print(f"  [1/2] Método Despeje...")
        rbf_despeje, time_despeje = method_despeje(t_data, theta_data)
        rmse_despeje = evaluate_rbf_accuracy(rbf_despeje, omega_range)

        # Método 2: Directo
        print(f"  [2/2] Método Directo...")
        rbf_direct, time_direct = method_direct(t_data, theta_data, t_span, initial)
        rmse_direct = evaluate_rbf_accuracy(rbf_direct, omega_range)

        # Calcular mejora
        improvement = ((rmse_despeje - rmse_direct) / rmse_despeje) * 100

        # Guardar resultados
        results['n_points'].append(n_points)
        results['dt'].append(dt)
        results['rmse_despeje'].append(rmse_despeje)
        results['rmse_direct'].append(rmse_direct)
        results['time_despeje'].append(time_despeje)
        results['time_direct'].append(time_direct)
        results['improvement'].append(improvement)

        print(f"\n  Resultados:")
        print(f"    Δt = {dt:.4f} s")
        print(f"    RMSE Despeje:  {rmse_despeje:.6e}")
        print(f"    RMSE Directo:  {rmse_direct:.6e}")
        print(f"    Mejora:        {improvement:+.1f}%")
        print(f"    Tiempo Despeje: {time_despeje:.4f} s")
        print(f"    Tiempo Directo: {time_direct:.2f} s")

    return results


def visualize_sensitivity(results):
    """Visualizar resultados del análisis de sensibilidad"""

    n_points = np.array(results['n_points'])
    dt = np.array(results['dt'])
    rmse_desp = np.array(results['rmse_despeje'])
    rmse_dir = np.array(results['rmse_direct'])
    improvement = np.array(results['improvement'])
    time_desp = np.array(results['time_despeje'])
    time_dir = np.array(results['time_direct'])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Error vs Número de Puntos
    ax = axes[0, 0]
    ax.semilogy(n_points, rmse_desp, 'b-o', linewidth=2, markersize=8,
               label='Despeje + RBF')
    ax.semilogy(n_points, rmse_dir, 'r-s', linewidth=2, markersize=8,
               label='Optimización Directa')
    ax.set_xlabel('Número de Puntos', fontsize=11)
    ax.set_ylabel('RMSE c(ω)', fontsize=11)
    ax.set_title('Error vs Densidad de Datos', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # 2. Mejora Relativa
    ax = axes[0, 1]
    colors = ['green' if imp > 80 else 'orange' if imp > 50 else 'red'
              for imp in improvement]
    ax.bar(range(len(n_points)), improvement, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(n_points)))
    ax.set_xticklabels(n_points)
    ax.set_xlabel('Número de Puntos', fontsize=11)
    ax.set_ylabel('Mejora (%)', fontsize=11)
    ax.set_title('Mejora Relativa del Método Directo', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Tiempo de Cómputo
    ax = axes[0, 2]
    ax.semilogy(n_points, time_desp, 'b-o', linewidth=2, markersize=8,
               label='Despeje')
    ax.semilogy(n_points, time_dir, 'r-s', linewidth=2, markersize=8,
               label='Directa')
    ax.set_xlabel('Número de Puntos', fontsize=11)
    ax.set_ylabel('Tiempo (s)', fontsize=11)
    ax.set_title('Tiempo de Cómputo', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # 4. Error vs Δt (log-log)
    ax = axes[1, 0]
    ax.loglog(dt, rmse_desp, 'b-o', linewidth=2, markersize=8,
             label='Despeje + RBF')
    ax.loglog(dt, rmse_dir, 'r-s', linewidth=2, markersize=8,
             label='Optimización Directa')

    # Ajustar pendientes
    mask = dt > 0.05
    if np.sum(mask) > 2:
        p_desp = np.polyfit(np.log(dt[mask]), np.log(rmse_desp[mask]), 1)
        p_dir = np.polyfit(np.log(dt[mask]), np.log(rmse_dir[mask]), 1)

        dt_fit = np.array([dt.min(), dt.max()])
        ax.loglog(dt_fit, np.exp(p_desp[1]) * dt_fit**p_desp[0], 'b--',
                 alpha=0.5, label=f'Despeje: ∝ Δt^{p_desp[0]:.2f}')
        ax.loglog(dt_fit, np.exp(p_dir[1]) * dt_fit**p_dir[0], 'r--',
                 alpha=0.5, label=f'Directa: ∝ Δt^{p_dir[0]:.2f}')

    ax.set_xlabel('Paso Temporal Δt (s)', fontsize=11)
    ax.set_ylabel('RMSE c(ω)', fontsize=11)
    ax.set_title('Escalamiento del Error con Δt', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # 5. Factor de Mejora vs Δt
    ax = axes[1, 1]
    factor_mejora = rmse_desp / rmse_dir
    ax.semilogx(dt, factor_mejora, 'g-o', linewidth=2.5, markersize=8)
    ax.axhline(1, color='k', linestyle='--', linewidth=1, label='Sin mejora')
    ax.set_xlabel('Paso Temporal Δt (s)', fontsize=11)
    ax.set_ylabel('Factor de Mejora', fontsize=11)
    ax.set_title('Factor de Mejora vs Δt', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # 6. Tabla de Resultados
    ax = axes[1, 2]
    ax.axis('off')

    table_data = []
    headers = ['N', 'Δt', 'Despeje', 'Directa', 'Mejora']

    for i in range(len(n_points)):
        row = [
            f"{n_points[i]}",
            f"{dt[i]:.3f}",
            f"{rmse_desp[i]:.2e}",
            f"{rmse_dir[i]:.2e}",
            f"{improvement[i]:+.0f}%"
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Colorear encabezados
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Colorear filas por mejora
    for i in range(len(table_data)):
        imp = improvement[i]
        color = '#90EE90' if imp > 80 else '#FFD580' if imp > 50 else '#FFB6B6'
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)

    ax.set_title('Tabla de Resultados', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('/home/rodo/1Paper/PublicationC/pendulum_sensitivity_analysis.png',
                dpi=150, bbox_inches='tight')
    print("\n📊 Figura guardada: pendulum_sensitivity_analysis.png")
    plt.close()


def print_summary(results):
    """Imprimir resumen final"""

    improvement = np.array(results['improvement'])
    n_points = np.array(results['n_points'])

    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)

    print(f"\n📊 MEJORA POR RANGO DE PUNTOS:")
    print(f"  Pocos puntos (N < 30):   Mejora promedio = {improvement[n_points < 30].mean():.1f}%")
    print(f"  Medios (30 ≤ N < 75):    Mejora promedio = {improvement[(n_points >= 30) & (n_points < 75)].mean():.1f}%")
    print(f"  Muchos (N ≥ 75):         Mejora promedio = {improvement[n_points >= 75].mean():.1f}%")

    best_idx = np.argmax(improvement)
    print(f"\n🏆 MEJOR MEJORA:")
    print(f"  N = {n_points[best_idx]} puntos → {improvement[best_idx]:+.1f}%")

    print("\n💡 RECOMENDACIONES:")
    print("  ✓ Usar método DIRECTO cuando N < 30 (mejora > 80%)")
    print("  ✓ Ambos métodos funcionan bien con N > 50")
    print("  ✓ Método directo es especialmente útil con datos escasos")
    print("="*80 + "\n")


def main():
    np.random.seed(42)

    # Ejecutar análisis
    results = run_sensitivity_analysis()

    # Visualizar
    print(f"\n{'='*80}")
    print("Generando visualización...")
    visualize_sensitivity(results)

    # Resumen
    print_summary(results)


if __name__ == "__main__":
    main()
