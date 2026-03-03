"""
IDENTIFICACIÓN DE FUERZA DE RESTITUCIÓN EN OSCILADOR DE DUFFING

Problema: Oscilador de Duffing forzado con amortiguamiento

  y'' + d·y' + f(y) = A·cos(ω·t)

donde:
  - d: coeficiente de amortiguamiento (conocido)
  - f(y) = a·y + b·y³: fuerza de restitución no lineal (DESCONOCIDA)
  - A·cos(ω·t): forzamiento externo periódico

Objetivo: Identificar f(y) usando RBF a partir de mediciones (t_i, y_i)

Métodos:
  1. Despeje + RBF: Calcular y' y y'' numéricamente, despejar f(y)
  2. Optimización Directa: Optimizar RBF dentro de la ecuación
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time


# Parámetros del sistema (algunos conocidos, otros a identificar)
D = 0.2          # Amortiguamiento (conocido)
A = 0.8          # Amplitud del forzamiento (conocida)
OMEGA = 1.2      # Frecuencia del forzamiento (conocida)

# Parámetros de f(y) = a·y + b·y³ (DESCONOCIDOS - a identificar)
A_SPRING = 1.0   # Coeficiente lineal
B_SPRING = 0.5   # Coeficiente cúbico


def true_restoring_force(y):
    """
    Fuerza de restitución REAL (desconocida en la práctica)
    f(y) = a·y + b·y³

    - Término lineal: resorte lineal (a > 0: resorte duro)
    - Término cúbico: no linealidad (b > 0: hardening spring)
    """
    return A_SPRING * y + B_SPRING * y**3


def forcing_term(t):
    """Forzamiento externo"""
    return A * np.cos(OMEGA * t)


def generate_experimental_data(t_span, initial_conditions, n_points, noise=0.0):
    """
    Generar datos experimentales del oscilador de Duffing

    Sistema:
      y' = v
      v' = A·cos(ω·t) - d·v - f(y)
    """
    def duffing_real(t, state):
        y, v = state
        dy_dt = v
        dv_dt = forcing_term(t) - D * v - true_restoring_force(y)
        return [dy_dt, dv_dt]

    # Integrar con alta precisión
    sol = solve_ivp(
        duffing_real,
        t_span,
        initial_conditions,
        t_eval=np.linspace(t_span[0], t_span[1], n_points),
        method='RK45',
        rtol=1e-10,
        atol=1e-12,
        max_step=0.05
    )

    t_data = sol.t
    y_data = sol.y[0]
    v_data = sol.y[1]

    # Agregar ruido experimental
    if noise > 0:
        y_data += np.random.normal(0, noise, len(y_data))
        v_data += np.random.normal(0, noise * 2, len(v_data))

    return t_data, y_data, v_data


class RBFOptimizable:
    """RBF con parámetros optimizables para aproximar f(y)"""

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

    def __call__(self, y):
        """Evaluar f(y) = RBF(y)"""
        y = np.atleast_1d(y)
        distances = np.abs(y.reshape(-1, 1) - self.centers.reshape(1, -1))
        phi = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))
        result = phi @ self.weights[:-1] + self.weights[-1]
        return result if len(y) > 1 else float(result[0])


def method_1_despeje(t_data, y_data):
    """
    MÉTODO 1: Despeje + RBF

    Calcular derivadas numéricas y despejar f(y)
    """
    print(f"\n{'='*70}")
    print(f"MÉTODO 1: Despeje + RBF (Método Tradicional)")
    print(f"{'='*70}")

    start_time = time.time()

    # Calcular derivadas numéricas
    y_prime = np.gradient(y_data, t_data)      # Primera derivada
    y_second = np.gradient(y_prime, t_data)    # Segunda derivada

    # Despejar f(y) de la ecuación:
    # y'' + d·y' + f(y) = A·cos(ω·t)
    # f(y) = A·cos(ω·t) - y'' - d·y'
    forcing = forcing_term(t_data)
    f_data = forcing - y_second - D * y_prime

    # Entrenar RBF: f(y) ≈ RBF(y)
    n_centers = 10
    y_min, y_max = y_data.min(), y_data.max()
    centers = np.linspace(y_min - 0.1, y_max + 0.1, n_centers)
    sigma = (y_max - y_min) / (2 * n_centers)

    # Calcular matriz de diseño
    distances = cdist(y_data.reshape(-1, 1), centers.reshape(-1, 1))
    phi = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    phi = np.hstack([phi, np.ones((phi.shape[0], 1))])

    # Resolver mínimos cuadrados
    weights = np.linalg.lstsq(phi, f_data, rcond=None)[0]

    # Crear RBF
    rbf = RBFOptimizable(n_centers)
    rbf.centers = centers
    rbf.sigma = sigma
    rbf.weights = weights

    elapsed = time.time() - start_time

    # Evaluar error en la función de restitución
    f_pred = np.array([rbf(y_val) for y_val in y_data])
    rmse_data = np.sqrt(np.mean((f_pred - f_data) ** 2))

    # Error comparado con función real
    f_true = true_restoring_force(y_data)
    rmse_true = np.sqrt(np.mean((f_pred - f_true) ** 2))

    print(f"  Centros RBF: {n_centers}")
    print(f"  Sigma: {sigma:.4f}")
    print(f"  Tiempo: {elapsed:.4f} s")
    print(f"  RMSE vs datos identificados: {rmse_data:.6e}")
    print(f"  RMSE vs fuerza real: {rmse_true:.6e}")

    return rbf, y_prime, f_data, elapsed


def method_2_direct_optimization(t_data, y_data, initial_conditions, t_span):
    """
    MÉTODO 2: Optimización Directa

    Optimizar parámetros de RBF dentro de la ODE
    """
    print(f"\n{'='*70}")
    print(f"MÉTODO 2: Optimización Directa (sin derivadas)")
    print(f"{'='*70}")

    n_centers = 5

    def solve_duffing_with_rbf(rbf, t_eval):
        """Resolver oscilador de Duffing con RBF como fuerza de restitución"""
        def duffing_rbf(t, state):
            y, v = state
            dy_dt = v
            dv_dt = forcing_term(t) - D * v - rbf(y)
            return [dy_dt, dv_dt]

        try:
            sol = solve_ivp(
                duffing_rbf,
                t_span,
                initial_conditions,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-5,
                atol=1e-7,
                max_step=0.1
            )

            if sol.success:
                return sol.y[0], True  # Retornar y
            else:
                return np.full_like(t_eval, np.nan), False
        except:
            return np.full_like(t_eval, np.nan), False

    def objective_function(params, rbf, t_data, y_data):
        """Función objetivo: ||y_ODE - y_data||²"""
        rbf.set_parameters(params)
        y_pred, success = solve_duffing_with_rbf(rbf, t_data)

        if not success or np.any(np.isnan(y_pred)):
            return 1e10

        return np.mean((y_pred - y_data) ** 2)

    # Inicialización
    rbf = RBFOptimizable(n_centers)

    y_min, y_max = y_data.min(), y_data.max()

    centers_init = np.linspace(y_min - 0.2, y_max + 0.2, n_centers)
    sigma_init = (y_max - y_min) / (2 * n_centers)
    weights_init = np.random.randn(n_centers + 1) * 0.1

    params_init = np.concatenate([centers_init, [sigma_init], weights_init])

    # Bounds
    bounds = []
    for _ in range(n_centers):
        bounds.append((y_min - 0.5, y_max + 0.5))
    bounds.append((0.01, 2.0))
    for _ in range(n_centers + 1):
        bounds.append((-10, 10))

    print(f"  Centros RBF: {n_centers}")
    print(f"  Parámetros totales: {len(params_init)}")
    print(f"  Optimizando con L-BFGS-B...")

    start_time = time.time()

    result = minimize(
        objective_function,
        params_init,
        args=(rbf, t_data, y_data),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 50, 'disp': False}
    )

    elapsed = time.time() - start_time

    rbf.set_parameters(result.x)
    y_pred, success = solve_duffing_with_rbf(rbf, t_data)

    rmse_y = np.sqrt(np.mean((y_pred - y_data) ** 2))

    print(f"  Éxito: {result.success}")
    print(f"  Iteraciones: {result.nit}")
    print(f"  Evaluaciones: {result.nfev}")
    print(f"  Tiempo: {elapsed:.2f} s")
    print(f"  RMSE y: {rmse_y:.6e}")

    return rbf, y_pred, elapsed


def visualize_results(t_data, y_data, v_data, rbf_despeje,
                     y_prime, f_data, rbf_direct, y_pred_direct):
    """Visualizar comparación de métodos"""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Datos experimentales: y(t)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_data, y_data, 'b-', linewidth=2, label='y(t)')
    ax1.set_xlabel('Tiempo (s)', fontsize=11)
    ax1.set_ylabel('Posición y', fontsize=11)
    ax1.set_title('Datos Experimentales: y(t)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Velocidad v(t)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_data, v_data, 'r-', linewidth=2, label='v(t) real')
    ax2.plot(t_data, y_prime, 'g--', linewidth=1.5, label='v ≈ dy/dt (numérico)')
    ax2.set_xlabel('Tiempo (s)', fontsize=11)
    ax2.set_ylabel('Velocidad v', fontsize=11)
    ax2.set_title('Velocidad', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Espacio de fases
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(y_data, v_data, 'b-', linewidth=2, alpha=0.7)
    ax3.scatter(y_data[0], v_data[0], s=200, c='green', marker='o',
               edgecolors='black', linewidths=2, label='Inicio', zorder=10)
    ax3.scatter(y_data[-1], v_data[-1], s=200, c='red', marker='s',
               edgecolors='black', linewidths=2, label='Final', zorder=10)
    ax3.set_xlabel('y', fontsize=11)
    ax3.set_ylabel('v', fontsize=11)
    ax3.set_title('Retrato de Fase', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Forzamiento externo
    ax4 = fig.add_subplot(gs[1, 0])
    t_fine = np.linspace(t_data[0], t_data[-1], 500)
    ax4.plot(t_fine, forcing_term(t_fine), 'purple', linewidth=2)
    ax4.scatter(t_data, forcing_term(t_data), c='purple', s=30, alpha=0.5, zorder=10)
    ax4.set_xlabel('Tiempo (s)', fontsize=11)
    ax4.set_ylabel(f'F(t) = {A}·cos({OMEGA}·t)', fontsize=11)
    ax4.set_title('Forzamiento Externo', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Fuerza identificada - Método 1 (Despeje)
    y_range = np.linspace(y_data.min() - 0.3, y_data.max() + 0.3, 200)
    f_true_range = true_restoring_force(y_range)
    f_rbf_despeje = np.array([rbf_despeje(y_val) for y_val in y_range])

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(y_range, f_true_range, 'k-', linewidth=3,
            label=f'Real: f(y) = {A_SPRING}y + {B_SPRING}y³', zorder=10)
    ax5.plot(y_range, f_rbf_despeje, 'b--', linewidth=2.5,
            label='RBF (despeje)', alpha=0.8)
    ax5.scatter(rbf_despeje.centers, [rbf_despeje(c) for c in rbf_despeje.centers],
               s=120, c='blue', marker='X', edgecolors='black', linewidths=2,
               label='Centros RBF', zorder=15)

    rmse_despeje = np.sqrt(np.mean((f_rbf_despeje - f_true_range) ** 2))
    ax5.set_title(f'Método 1: Despeje + RBF\nRMSE = {rmse_despeje:.6e}',
                 fontsize=11, fontweight='bold')
    ax5.set_xlabel('y', fontsize=11)
    ax5.set_ylabel('Fuerza f(y)', fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. Fuerza identificada - Método 2 (Optimización Directa)
    f_rbf_direct = np.array([rbf_direct(y_val) for y_val in y_range])

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(y_range, f_true_range, 'k-', linewidth=3,
            label=f'Real: f(y) = {A_SPRING}y + {B_SPRING}y³', zorder=10)
    ax6.plot(y_range, f_rbf_direct, 'r--', linewidth=2.5,
            label='RBF (optimización)', alpha=0.8)
    ax6.scatter(rbf_direct.centers, [rbf_direct(c) for c in rbf_direct.centers],
               s=120, c='red', marker='X', edgecolors='black', linewidths=2,
               label='Centros RBF', zorder=15)

    rmse_direct = np.sqrt(np.mean((f_rbf_direct - f_true_range) ** 2))
    ax6.set_title(f'Método 2: Optimización Directa\nRMSE = {rmse_direct:.6e}',
                 fontsize=11, fontweight='bold')
    ax6.set_xlabel('y', fontsize=11)
    ax6.set_ylabel('Fuerza f(y)', fontsize=11)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # 7. Comparación de fuerzas
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(y_range, f_true_range, 'k-', linewidth=3, label='Real', zorder=10)
    ax7.plot(y_range, f_rbf_despeje, 'b--', linewidth=2,
            label=f'Despeje (RMSE={rmse_despeje:.4f})', alpha=0.7)
    ax7.plot(y_range, f_rbf_direct, 'r--', linewidth=2,
            label=f'Directa (RMSE={rmse_direct:.4f})', alpha=0.7)

    mejora = ((rmse_despeje - rmse_direct) / rmse_despeje) * 100
    ax7.set_title(f'Comparación de Fuerzas\nMejora: {mejora:+.1f}%',
                 fontsize=11, fontweight='bold')
    ax7.set_xlabel('y', fontsize=11)
    ax7.set_ylabel('f(y)', fontsize=11)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    # 8. Validación: y(t) con RBF despeje
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t_data, y_data, 'ko', markersize=5, label='Datos', zorder=10)

    # Resolver ODE con RBF de despeje
    def duffing_despeje(t, state):
        y, v = state
        return [v, forcing_term(t) - D * v - rbf_despeje(y)]

    sol_despeje = solve_ivp(duffing_despeje, (t_data[0], t_data[-1]),
                           [y_data[0], v_data[0]], t_eval=t_data,
                           method='RK45', rtol=1e-6, atol=1e-8, max_step=0.05)

    ax8.plot(t_data, sol_despeje.y[0], 'b-', linewidth=2, label='RBF despeje')
    rmse_val_despeje = np.sqrt(np.mean((sol_despeje.y[0] - y_data) ** 2))
    ax8.set_title(f'Validación Despeje\nRMSE = {rmse_val_despeje:.6e}',
                 fontsize=11, fontweight='bold')
    ax8.set_xlabel('Tiempo (s)', fontsize=11)
    ax8.set_ylabel('y', fontsize=11)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Validación: y(t) con RBF directa
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(t_data, y_data, 'ko', markersize=5, label='Datos', zorder=10)
    ax9.plot(t_data, y_pred_direct, 'r-', linewidth=2, label='RBF directa')

    rmse_val_direct = np.sqrt(np.mean((y_pred_direct - y_data) ** 2))
    ax9.set_title(f'Validación Optimización\nRMSE = {rmse_val_direct:.6e}',
                 fontsize=11, fontweight='bold')
    ax9.set_xlabel('Tiempo (s)', fontsize=11)
    ax9.set_ylabel('y', fontsize=11)
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.savefig('/home/rodo/1Paper/PublicationD/duffing_rbf_identification.png',
                dpi=150, bbox_inches='tight')
    print("\n📊 Figura guardada: duffing_rbf_identification.png")
    plt.close()


def main():
    """Programa principal"""

    print("\n" + "="*70)
    print("IDENTIFICACIÓN DE FUERZA EN OSCILADOR DE DUFFING")
    print("="*70)
    print(f"\nEcuación: y'' + {D}·y' + f(y) = {A}·cos({OMEGA}·t)")
    print(f"Fuerza real: f(y) = {A_SPRING}·y + {B_SPRING}·y³")

    # Configuración del experimento
    t_span = (0, 15)  # Tiempo suficiente para ver varios ciclos
    y0 = 0.5
    v0 = 0.0
    initial_conditions = [y0, v0]

    n_points = 40
    noise_level = 0.0

    print(f"\n📋 CONFIGURACIÓN:")
    print(f"  Tiempo: t ∈ [{t_span[0]}, {t_span[1]}] s")
    print(f"  Condiciones iniciales: y₀ = {y0}, v₀ = {v0}")
    print(f"  Puntos de medición: {n_points}")
    print(f"  Ruido: {noise_level}")

    # Generar datos experimentales
    print(f"\n{'─'*70}")
    print("Generando datos experimentales...")
    t_data, y_data, v_data = generate_experimental_data(
        t_span, initial_conditions, n_points, noise_level
    )
    print(f"  ✓ {len(t_data)} mediciones")
    print(f"  Rango y: [{y_data.min():.4f}, {y_data.max():.4f}]")
    print(f"  Rango v: [{v_data.min():.4f}, {v_data.max():.4f}]")

    # MÉTODO 1: Despeje + RBF
    rbf_despeje, y_prime, f_data, time_despeje = method_1_despeje(
        t_data, y_data
    )

    # MÉTODO 2: Optimización Directa
    rbf_direct, y_pred_direct, time_direct = method_2_direct_optimization(
        t_data, y_data, initial_conditions, t_span
    )

    # Visualización
    print(f"\n{'='*70}")
    print("Generando visualización...")
    visualize_results(t_data, y_data, v_data, rbf_despeje,
                     y_prime, f_data, rbf_direct, y_pred_direct)

    # Resumen final
    print(f"\n{'='*70}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*70}")

    y_test = np.linspace(y_data.min(), y_data.max(), 100)
    f_true = true_restoring_force(y_test)
    f_pred_despeje = np.array([rbf_despeje(y_val) for y_val in y_test])
    f_pred_direct = np.array([rbf_direct(y_val) for y_val in y_test])

    rmse_despeje = np.sqrt(np.mean((f_pred_despeje - f_true) ** 2))
    rmse_direct = np.sqrt(np.mean((f_pred_direct - f_true) ** 2))
    mejora = ((rmse_despeje - rmse_direct) / rmse_despeje) * 100

    print(f"\n{'Método':<30} {'RMSE f(y)':>15} {'Tiempo':>15}")
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
