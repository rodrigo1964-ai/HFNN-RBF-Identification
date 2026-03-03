"""
IDENTIFICACIÓN DE FRICCIÓN VISCOSA EN PÉNDULO CON RBF

Problema físico:
  Péndulo con fricción viscosa de ley desconocida

  Ecuación: θ'' + c(θ') + ω₀²·sin(θ) = 0

  Objetivo: Identificar c(θ') usando RBF a partir de mediciones (t_i, θ_i)

Método 1: Despeje + RBF
  - Calcular θ' y θ'' numéricamente
  - Despejar: c(θ') = -θ'' - ω₀²·sin(θ)
  - Entrenar RBF(θ') ≈ c(θ')

Método 2: Optimización Directa
  - Parametrizar c(θ') = RBF(θ'; params)
  - Resolver ODE con RBF
  - Minimizar ||θ_ODE - θ_data||²
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import time


# Parámetros físicos del péndulo
OMEGA_0 = 2.0  # Frecuencia natural (rad/s) - conocida
B1 = 0.3       # Coeficiente lineal de fricción - DESCONOCIDO
B2 = 0.05      # Coeficiente cúbico de fricción - DESCONOCIDO


def true_friction(omega):
    """
    Ley de fricción REAL (desconocida en la práctica)
    c(ω) = b₁·ω + b₂·ω³

    Fricción viscosa no lineal realista
    """
    return B1 * omega + B2 * omega**3


def generate_experimental_data(t_span, initial_conditions, n_points, noise=0.0):
    """
    Generar datos experimentales del péndulo

    Sistema:
      θ' = ω
      ω' = -ω₀²·sin(θ) - c(ω)
    """
    def pendulum_real(t, state):
        theta, omega = state
        dtheta_dt = omega
        domega_dt = -OMEGA_0**2 * np.sin(theta) - true_friction(omega)
        return [dtheta_dt, domega_dt]

    # Integrar con alta precisión
    sol = solve_ivp(
        pendulum_real,
        t_span,
        initial_conditions,
        t_eval=np.linspace(t_span[0], t_span[1], n_points),
        method='RK45',
        rtol=1e-10,
        atol=1e-12
    )

    t_data = sol.t
    theta_data = sol.y[0]
    omega_data = sol.y[1]

    # Agregar ruido experimental
    if noise > 0:
        theta_data += np.random.normal(0, noise, len(theta_data))
        omega_data += np.random.normal(0, noise * 2, len(omega_data))

    return t_data, theta_data, omega_data


class RBFOptimizable:
    """RBF con parámetros optimizables para aproximar c(ω)"""

    def __init__(self, n_centers):
        self.n_centers = n_centers
        self.centers = None
        self.sigma = None
        self.weights = None

    def set_parameters(self, params):
        """params = [c1, ..., cM, sigma, w1, ..., wM, w0]"""
        n = self.n_centers
        self.centers = params[:n]
        self.sigma = params[n]
        self.weights = params[n+1:]

    def __call__(self, omega):
        """Evaluar c(ω) = RBF(ω)"""
        omega = np.atleast_1d(omega)
        distances = np.abs(omega.reshape(-1, 1) - self.centers.reshape(1, -1))
        phi = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))
        result = phi @ self.weights[:-1] + self.weights[-1]
        return result if len(omega) > 1 else float(result[0])


def method_1_despeje(t_data, theta_data):
    """
    MÉTODO 1: Despeje + RBF

    Calcular derivadas numéricas y despejar c(θ')
    """
    print(f"\n{'='*70}")
    print(f"MÉTODO 1: Despeje + RBF (Método Tradicional)")
    print(f"{'='*70}")

    start_time = time.time()

    # Calcular derivadas numéricas
    theta_prime = np.gradient(theta_data, t_data)  # Primera derivada
    theta_second = np.gradient(theta_prime, t_data)  # Segunda derivada

    # Despejar c(θ') de la ecuación:
    # θ'' + c(θ') + ω₀²·sin(θ) = 0
    # c(θ') = -θ'' - ω₀²·sin(θ)
    c_data = -theta_second - OMEGA_0**2 * np.sin(theta_data)

    # Entrenar RBF: c(θ') ≈ RBF(θ')
    n_centers = 8
    omega_min, omega_max = theta_prime.min(), theta_prime.max()
    centers = np.linspace(omega_min - 0.1, omega_max + 0.1, n_centers)
    sigma = (omega_max - omega_min) / (2 * n_centers)

    # Calcular matriz de diseño
    distances = cdist(theta_prime.reshape(-1, 1), centers.reshape(-1, 1))
    phi = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    phi = np.hstack([phi, np.ones((phi.shape[0], 1))])

    # Resolver mínimos cuadrados
    weights = np.linalg.lstsq(phi, c_data, rcond=None)[0]

    # Crear RBF
    rbf = RBFOptimizable(n_centers)
    rbf.centers = centers
    rbf.sigma = sigma
    rbf.weights = weights

    elapsed = time.time() - start_time

    # Evaluar error en la función de fricción
    c_pred = np.array([rbf(omega) for omega in theta_prime])
    rmse_friction = np.sqrt(np.mean((c_pred - c_data) ** 2))

    # Error comparado con función real
    c_true = true_friction(theta_prime)
    rmse_true = np.sqrt(np.mean((c_pred - c_true) ** 2))

    print(f"  Centros RBF: {n_centers}")
    print(f"  Sigma: {sigma:.4f}")
    print(f"  Tiempo: {elapsed:.4f} s")
    print(f"  RMSE vs datos identificados: {rmse_friction:.6e}")
    print(f"  RMSE vs fricción real: {rmse_true:.6e}")

    return rbf, theta_prime, c_data, elapsed


def method_2_direct_optimization(t_data, theta_data, initial_conditions, t_span):
    """
    MÉTODO 2: Optimización Directa

    Optimizar parámetros de RBF dentro de la ODE
    """
    print(f"\n{'='*70}")
    print(f"MÉTODO 2: Optimización Directa (sin derivadas)")
    print(f"{'='*70}")

    n_centers = 6

    def solve_pendulum_with_rbf(rbf, t_eval):
        """Resolver péndulo con RBF como fricción"""
        def pendulum_rbf(t, state):
            theta, omega = state
            dtheta_dt = omega
            domega_dt = -OMEGA_0**2 * np.sin(theta) - rbf(omega)
            return [dtheta_dt, domega_dt]

        try:
            sol = solve_ivp(
                pendulum_rbf,
                t_span,
                initial_conditions,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-6,
                atol=1e-8
            )

            if sol.success:
                return sol.y[0], True  # Retornar θ
            else:
                return np.full_like(t_eval, np.nan), False
        except:
            return np.full_like(t_eval, np.nan), False

    def objective_function(params, rbf, t_data, theta_data):
        """Función objetivo: ||θ_ODE - θ_data||²"""
        rbf.set_parameters(params)
        theta_pred, success = solve_pendulum_with_rbf(rbf, t_data)

        if not success or np.any(np.isnan(theta_pred)):
            return 1e10

        return np.mean((theta_pred - theta_data) ** 2)

    # Inicialización
    rbf = RBFOptimizable(n_centers)

    omega_estimate = np.gradient(theta_data, t_data)
    omega_min, omega_max = omega_estimate.min(), omega_estimate.max()

    centers_init = np.linspace(omega_min - 0.2, omega_max + 0.2, n_centers)
    sigma_init = (omega_max - omega_min) / (2 * n_centers)
    weights_init = np.random.randn(n_centers + 1) * 0.1

    params_init = np.concatenate([centers_init, [sigma_init], weights_init])

    # Bounds
    bounds = []
    for _ in range(n_centers):
        bounds.append((omega_min - 0.5, omega_max + 0.5))
    bounds.append((0.01, 2.0))
    for _ in range(n_centers + 1):
        bounds.append((-5, 5))

    print(f"  Centros RBF: {n_centers}")
    print(f"  Parámetros totales: {len(params_init)}")
    print(f"  Optimizando con L-BFGS-B...")

    start_time = time.time()

    result = minimize(
        objective_function,
        params_init,
        args=(rbf, t_data, theta_data),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 150, 'disp': False}
    )

    elapsed = time.time() - start_time

    rbf.set_parameters(result.x)
    theta_pred, success = solve_pendulum_with_rbf(rbf, t_data)

    rmse_theta = np.sqrt(np.mean((theta_pred - theta_data) ** 2))

    print(f"  Éxito: {result.success}")
    print(f"  Iteraciones: {result.nit}")
    print(f"  Evaluaciones: {result.nfev}")
    print(f"  Tiempo: {elapsed:.2f} s")
    print(f"  RMSE θ: {rmse_theta:.6e}")

    return rbf, theta_pred, elapsed


def visualize_results(t_data, theta_data, omega_data, rbf_despeje,
                     theta_prime, c_data, rbf_direct, theta_pred_direct):
    """Visualizar comparación de métodos"""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Datos experimentales: θ(t) y ω(t)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_data, theta_data, 'b-', linewidth=2, label='θ(t)')
    ax1.set_xlabel('Tiempo (s)', fontsize=11)
    ax1.set_ylabel('Ángulo θ (rad)', fontsize=11)
    ax1.set_title('Datos Experimentales: θ(t)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_data, omega_data, 'r-', linewidth=2, label='ω(t) real')
    ax2.plot(t_data, theta_prime, 'g--', linewidth=1.5, label='ω ≈ dθ/dt (numérico)')
    ax2.set_xlabel('Tiempo (s)', fontsize=11)
    ax2.set_ylabel('Velocidad ω (rad/s)', fontsize=11)
    ax2.set_title('Velocidad Angular', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 2. Espacio de fases
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(theta_data, omega_data, 'b-', linewidth=2, alpha=0.7)
    ax3.scatter(theta_data[0], omega_data[0], s=200, c='green', marker='o',
               edgecolors='black', linewidths=2, label='Inicio', zorder=10)
    ax3.scatter(theta_data[-1], omega_data[-1], s=200, c='red', marker='s',
               edgecolors='black', linewidths=2, label='Final', zorder=10)
    ax3.set_xlabel('θ (rad)', fontsize=11)
    ax3.set_ylabel('ω (rad/s)', fontsize=11)
    ax3.set_title('Retrato de Fase', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 3. Fricción identificada - Método 1 (Despeje)
    omega_range = np.linspace(omega_data.min() - 0.2, omega_data.max() + 0.2, 200)
    c_true_range = true_friction(omega_range)
    c_rbf_despeje = np.array([rbf_despeje(w) for w in omega_range])

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(omega_range, c_true_range, 'k-', linewidth=3,
            label=f'Real: c(ω) = {B1}ω + {B2}ω³', zorder=10)
    ax4.plot(omega_range, c_rbf_despeje, 'b--', linewidth=2.5,
            label='RBF (despeje)', alpha=0.8)
    ax4.scatter(rbf_despeje.centers, [rbf_despeje(c) for c in rbf_despeje.centers],
               s=120, c='blue', marker='X', edgecolors='black', linewidths=2,
               label='Centros RBF', zorder=15)

    rmse_despeje = np.sqrt(np.mean((c_rbf_despeje - c_true_range) ** 2))
    ax4.set_title(f'Método 1: Despeje + RBF\nRMSE = {rmse_despeje:.6e}',
                 fontsize=11, fontweight='bold')
    ax4.set_xlabel('ω (rad/s)', fontsize=11)
    ax4.set_ylabel('Fricción c(ω)', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 4. Fricción identificada - Método 2 (Optimización Directa)
    c_rbf_direct = np.array([rbf_direct(w) for w in omega_range])

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(omega_range, c_true_range, 'k-', linewidth=3,
            label=f'Real: c(ω) = {B1}ω + {B2}ω³', zorder=10)
    ax5.plot(omega_range, c_rbf_direct, 'r--', linewidth=2.5,
            label='RBF (optimización)', alpha=0.8)
    ax5.scatter(rbf_direct.centers, [rbf_direct(c) for c in rbf_direct.centers],
               s=120, c='red', marker='X', edgecolors='black', linewidths=2,
               label='Centros RBF', zorder=15)

    rmse_direct = np.sqrt(np.mean((c_rbf_direct - c_true_range) ** 2))
    ax5.set_title(f'Método 2: Optimización Directa\nRMSE = {rmse_direct:.6e}',
                 fontsize=11, fontweight='bold')
    ax5.set_xlabel('ω (rad/s)', fontsize=11)
    ax5.set_ylabel('Fricción c(ω)', fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 5. Comparación de fricciones
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(omega_range, c_true_range, 'k-', linewidth=3, label='Real', zorder=10)
    ax6.plot(omega_range, c_rbf_despeje, 'b--', linewidth=2,
            label=f'Despeje (RMSE={rmse_despeje:.4f})', alpha=0.7)
    ax6.plot(omega_range, c_rbf_direct, 'r--', linewidth=2,
            label=f'Directa (RMSE={rmse_direct:.4f})', alpha=0.7)

    mejora = ((rmse_despeje - rmse_direct) / rmse_despeje) * 100
    ax6.set_title(f'Comparación\nMejora: {mejora:+.1f}%',
                 fontsize=11, fontweight='bold')
    ax6.set_xlabel('ω (rad/s)', fontsize=11)
    ax6.set_ylabel('Fricción c(ω)', fontsize=11)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # 6. Validación: θ(t) con RBF despeje
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(t_data, theta_data, 'ko', markersize=5, label='Datos', zorder=10)

    # Resolver ODE con RBF de despeje
    def pendulum_despeje(t, state):
        theta, omega = state
        return [omega, -OMEGA_0**2 * np.sin(theta) - rbf_despeje(omega)]

    sol_despeje = solve_ivp(pendulum_despeje, (t_data[0], t_data[-1]),
                           [theta_data[0], omega_data[0]], t_eval=t_data,
                           method='RK45', rtol=1e-6, atol=1e-8)

    ax7.plot(t_data, sol_despeje.y[0], 'b-', linewidth=2, label='RBF despeje')
    rmse_val_despeje = np.sqrt(np.mean((sol_despeje.y[0] - theta_data) ** 2))
    ax7.set_title(f'Validación Despeje\nRMSE = {rmse_val_despeje:.6e}',
                 fontsize=11, fontweight='bold')
    ax7.set_xlabel('Tiempo (s)', fontsize=11)
    ax7.set_ylabel('θ (rad)', fontsize=11)
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 7. Validación: θ(t) con RBF directa
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t_data, theta_data, 'ko', markersize=5, label='Datos', zorder=10)
    ax8.plot(t_data, theta_pred_direct, 'r-', linewidth=2, label='RBF directa')

    rmse_val_direct = np.sqrt(np.mean((theta_pred_direct - theta_data) ** 2))
    ax8.set_title(f'Validación Optimización\nRMSE = {rmse_val_direct:.6e}',
                 fontsize=11, fontweight='bold')
    ax8.set_xlabel('Tiempo (s)', fontsize=11)
    ax8.set_ylabel('θ (rad)', fontsize=11)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 8. Errores
    ax9 = fig.add_subplot(gs[2, 2])
    error_despeje = np.abs(sol_despeje.y[0] - theta_data)
    error_direct = np.abs(theta_pred_direct - theta_data)

    ax9.semilogy(t_data, error_despeje, 'b-', linewidth=2, label='Despeje')
    ax9.semilogy(t_data, error_direct, 'r-', linewidth=2, label='Optimización')
    ax9.set_title('Error Absoluto |θ_pred - θ_data|', fontsize=11, fontweight='bold')
    ax9.set_xlabel('Tiempo (s)', fontsize=11)
    ax9.set_ylabel('Error absoluto', fontsize=11)
    ax9.legend()
    ax9.grid(True, alpha=0.3, which='both')

    plt.savefig('/home/rodo/1Paper/PublicationC/pendulum_rbf_identification.png',
                dpi=150, bbox_inches='tight')
    print("\n📊 Figura guardada: pendulum_rbf_identification.png")
    plt.close()


def main():
    """Programa principal"""

    print("\n" + "="*70)
    print("IDENTIFICACIÓN DE FRICCIÓN VISCOSA EN PÉNDULO CON RBF")
    print("="*70)
    print(f"\nFricción real: c(ω) = {B1}·ω + {B2}·ω³")
    print(f"Frecuencia natural: ω₀ = {OMEGA_0} rad/s")

    # Configuración del experimento
    t_span = (0, 15)
    theta0 = 0.8  # Ángulo inicial (rad) ~ 45 grados
    omega0 = 0.0  # Velocidad inicial
    initial_conditions = [theta0, omega0]

    n_points = 50
    noise_level = 0.0

    print(f"\n📋 CONFIGURACIÓN:")
    print(f"  Tiempo: t ∈ [{t_span[0]}, {t_span[1]}] s")
    print(f"  Condiciones iniciales: θ₀ = {theta0} rad, ω₀ = {omega0} rad/s")
    print(f"  Puntos de medición: {n_points}")
    print(f"  Ruido: {noise_level}")

    # Generar datos experimentales
    print(f"\n{'─'*70}")
    print("Generando datos experimentales...")
    t_data, theta_data, omega_data = generate_experimental_data(
        t_span, initial_conditions, n_points, noise_level
    )
    print(f"  ✓ {len(t_data)} mediciones")
    print(f"  Rango θ: [{theta_data.min():.4f}, {theta_data.max():.4f}] rad")
    print(f"  Rango ω: [{omega_data.min():.4f}, {omega_data.max():.4f}] rad/s")

    # MÉTODO 1: Despeje + RBF
    rbf_despeje, theta_prime, c_data, time_despeje = method_1_despeje(
        t_data, theta_data
    )

    # MÉTODO 2: Optimización Directa
    rbf_direct, theta_pred_direct, time_direct = method_2_direct_optimization(
        t_data, theta_data, initial_conditions, t_span
    )

    # Visualización
    print(f"\n{'='*70}")
    print("Generando visualización...")
    visualize_results(t_data, theta_data, omega_data, rbf_despeje,
                     theta_prime, c_data, rbf_direct, theta_pred_direct)

    # Resumen final
    print(f"\n{'='*70}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*70}")

    omega_test = np.linspace(omega_data.min(), omega_data.max(), 100)
    c_true = true_friction(omega_test)
    c_pred_despeje = np.array([rbf_despeje(w) for w in omega_test])
    c_pred_direct = np.array([rbf_direct(w) for w in omega_test])

    rmse_despeje = np.sqrt(np.mean((c_pred_despeje - c_true) ** 2))
    rmse_direct = np.sqrt(np.mean((c_pred_direct - c_true) ** 2))
    mejora = ((rmse_despeje - rmse_direct) / rmse_despeje) * 100

    print(f"\n{'Método':<30} {'RMSE c(ω)':>15} {'Tiempo':>15}")
    print("─"*70)
    print(f"{'Despeje + RBF':<30} {rmse_despeje:>15.6e} {time_despeje:>14.3f}s")
    print(f"{'Optimización Directa':<30} {rmse_direct:>15.6e} {time_direct:>14.2f}s")
    print("="*70)
    print(f"\n✨ Mejora en identificación: {mejora:+.1f}%")
    print(f"⏱️  Factor de tiempo: {time_direct/time_despeje:.0f}x más lento")

    print("\n" + "="*70)
    print("✓ Identificación completada exitosamente")
    print("="*70 + "\n")


if __name__ == "__main__":
    np.random.seed(42)
    main()
