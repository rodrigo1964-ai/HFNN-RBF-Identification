"""
Resolver ODE: y' + RBF(y) = sin(t)

Métodos implementados:
1. Integración numérica con RBF fija
2. Método de colocación con RBF
3. Optimización de parámetros RBF para satisfacer la ODE
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize
from scipy.spatial.distance import cdist


class RBFNetwork:
    """Red RBF simple"""

    def __init__(self, centers, sigma, weights):
        self.centers = np.array(centers).reshape(-1, 1)
        self.sigma = sigma
        self.weights = np.array(weights).reshape(-1, 1)

    def __call__(self, y):
        """Evaluar RBF(y)"""
        y = np.array(y).reshape(-1, 1)
        distances = cdist(y, self.centers, 'euclidean')
        phi = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        return (phi @ self.weights).flatten()

    def derivative(self, y):
        """Derivada de RBF respecto a y: d(RBF)/dy"""
        y = np.array(y).reshape(-1, 1)
        distances = cdist(y, self.centers, 'euclidean')
        phi = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

        # Derivada de cada gaussiana: d/dy exp(-dist^2/2sigma^2)
        # = -1/sigma^2 * (y - c) * exp(-dist^2/2sigma^2)
        d_phi = np.zeros_like(phi)
        for i in range(len(self.centers)):
            d_phi[:, i] = -(y.flatten() - self.centers[i, 0]) / (self.sigma ** 2) * phi[:, i]

        # No incluimos derivada del bias (es 0)
        return (d_phi @ self.weights[:-1]).flatten()


def method1_numerical_integration(rbf, t_span, y0):
    """
    Método 1: Integración numérica con RBF fija

    Resuelve: y' = sin(t) - RBF(y)
    """
    print("\n" + "="*70)
    print("MÉTODO 1: Integración Numérica (solve_ivp)")
    print("="*70)

    def ode_system(t, y):
        """y' = sin(t) - RBF(y)"""
        return np.sin(t) - rbf(y)

    # Resolver ODE
    sol = solve_ivp(
        ode_system,
        t_span,
        [y0],
        method='RK45',
        dense_output=True,
        max_step=0.01
    )

    print(f"  Estado: {sol.message}")
    print(f"  Número de evaluaciones: {sol.nfev}")
    print(f"  Puntos calculados: {len(sol.t)}")

    return sol


def method2_collocation(n_centers_rbf, sigma_rbf, t_span, y0, n_collocation=50):
    """
    Método 2: Método de colocación con RBF

    Aproximamos y(t) ≈ Σ w_i * φ_i(t)
    donde φ_i son RBFs centradas en el dominio temporal
    """
    print("\n" + "="*70)
    print("MÉTODO 2: Método de Colocación con RBF")
    print("="*70)

    t_min, t_max = t_span

    # Puntos de colocación en el dominio temporal
    t_collocation = np.linspace(t_min, t_max, n_collocation)

    # Centros para RBF en el dominio temporal
    t_centers = np.linspace(t_min, t_max, n_centers_rbf)

    print(f"  Puntos de colocación: {n_collocation}")
    print(f"  Centros RBF (temporal): {n_centers_rbf}")
    print(f"  Sigma RBF: {sigma_rbf}")

    def rbf_basis(t, centers, sigma):
        """Funciones de base RBF"""
        t = np.array(t).reshape(-1, 1)
        centers = np.array(centers).reshape(-1, 1)
        distances = cdist(t, centers, 'euclidean')
        return np.exp(-(distances ** 2) / (2 * sigma ** 2))

    def rbf_basis_derivative(t, centers, sigma):
        """Derivadas de las funciones de base RBF respecto a t"""
        t = np.array(t).reshape(-1, 1)
        centers = np.array(centers).reshape(-1, 1)
        distances = cdist(t, centers, 'euclidean')
        phi = np.exp(-(distances ** 2) / (2 * sigma ** 2))

        # d/dt exp(-dist^2/2sigma^2) = -1/sigma^2 * (t - c) * exp(...)
        d_phi = np.zeros_like(phi)
        for i in range(len(centers)):
            d_phi[:, i] = -(t.flatten() - centers[i, 0]) / (sigma ** 2) * phi[:, i]
        return d_phi

    # Construir matriz del sistema
    # y(t) = Φ(t) @ w + bias
    # y'(t) = Φ'(t) @ w
    # ODE: Φ'(t) @ w + RBF(Φ(t) @ w + bias) = sin(t)

    # Para simplificar, asumimos RBF lineal: RBF(y) ≈ a*y + b
    # Entonces: Φ'(t) @ w + a*(Φ(t) @ w + bias) + b = sin(t)
    # (Φ'(t) + a*Φ(t)) @ w = sin(t) - a*bias - b

    # Aproximación lineal: asumimos RBF(y) ≈ α*y
    alpha = 0.5  # coeficiente de aproximación lineal

    Phi = rbf_basis(t_collocation, t_centers, sigma_rbf)
    Phi_prime = rbf_basis_derivative(t_collocation, t_centers, sigma_rbf)

    # Añadir término de bias
    Phi = np.hstack([Phi, np.ones((Phi.shape[0], 1))])
    Phi_prime = np.hstack([Phi_prime, np.zeros((Phi_prime.shape[0], 1))])

    # Sistema: (Φ' + α*Φ) @ w = sin(t)
    A = Phi_prime + alpha * Phi
    b = np.sin(t_collocation)

    # Añadir condición inicial: y(t0) = y0
    Phi_0 = rbf_basis(np.array([t_min]), t_centers, sigma_rbf)
    Phi_0 = np.hstack([Phi_0, np.ones((1, 1))])

    # Sistema aumentado
    A_aug = np.vstack([Phi_0, A])
    b_aug = np.hstack([y0, b])

    # Resolver sistema de mínimos cuadrados
    weights, residuals, rank, s = np.linalg.lstsq(A_aug, b_aug, rcond=None)

    print(f"  Residuo: {np.linalg.norm(A_aug @ weights - b_aug):.6e}")

    # Función solución
    def solution(t):
        Phi_t = rbf_basis(t, t_centers, sigma_rbf)
        Phi_t = np.hstack([Phi_t, np.ones((Phi_t.shape[0] if len(Phi_t.shape) > 1 else 1, 1))])
        return Phi_t @ weights

    return solution, t_collocation


def method3_optimize_rbf_parameters(rbf_structure, t_span, y0, n_points=100):
    """
    Método 3: Optimizar parámetros de RBF para satisfacer la ODE

    Encuentra los pesos de la RBF tal que la solución de y' + RBF(y) = sin(t)
    satisfaga ciertas propiedades deseadas.
    """
    print("\n" + "="*70)
    print("MÉTODO 3: Optimización de Parámetros RBF")
    print("="*70)

    t_min, t_max = t_span
    t_train = np.linspace(t_min, t_max, n_points)

    centers = rbf_structure['centers']
    sigma = rbf_structure['sigma']
    n_centers = len(centers)

    print(f"  Puntos de entrenamiento: {n_points}")
    print(f"  Centros RBF: {n_centers}")

    def create_rbf(weights):
        return RBFNetwork(centers, sigma, weights)

    def residual_function(weights):
        """
        Calcula el residuo de la ODE en los puntos de entrenamiento
        Integramos numéricamente y calculamos el error
        """
        rbf = create_rbf(weights)

        # Resolver ODE con estos parámetros
        def ode_system(t, y):
            return np.sin(t) - rbf(y)

        sol = solve_ivp(ode_system, t_span, [y0], t_eval=t_train, method='RK45')

        if not sol.success:
            return np.ones(len(t_train)) * 1e10

        y_sol = sol.y[0]

        # Calcular residuo: y' + RBF(y) - sin(t)
        y_prime = np.gradient(y_sol, t_train)
        residuals = y_prime + rbf(y_sol) - np.sin(t_train)

        return residuals

    # Inicializar pesos aleatoriamente
    weights_init = np.random.randn(n_centers + 1) * 0.1

    print("  Optimizando parámetros...")
    result = least_squares(
        residual_function,
        weights_init,
        method='lm',
        verbose=0,
        max_nfev=100
    )

    print(f"  Estado: {result.message}")
    print(f"  Evaluaciones: {result.nfev}")
    print(f"  Residuo final: {np.linalg.norm(result.fun):.6e}")

    optimal_rbf = create_rbf(result.x)

    # Resolver ODE con RBF optimizada
    sol = method1_numerical_integration(optimal_rbf, t_span, y0)

    return optimal_rbf, sol


def visualize_solutions(solutions_dict, t_span):
    """Visualizar todas las soluciones"""

    t_plot = np.linspace(t_span[0], t_span[1], 500)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Gráfica 1: Todas las soluciones
    ax1 = axes[0, 0]
    colors = ['blue', 'red', 'green', 'purple']

    for idx, (name, data) in enumerate(solutions_dict.items()):
        if 'sol' in data:
            sol = data['sol']
            y_plot = sol.sol(t_plot) if hasattr(sol, 'sol') else None
            if y_plot is not None:
                ax1.plot(t_plot, y_plot[0], label=name, linewidth=2,
                        color=colors[idx % len(colors)])
        elif 'func' in data:
            y_plot = data['func'](t_plot.reshape(-1, 1)).flatten()
            ax1.plot(t_plot, y_plot, label=name, linewidth=2,
                    color=colors[idx % len(colors)], linestyle='--')

    ax1.set_xlabel('t', fontsize=12)
    ax1.set_ylabel('y(t)', fontsize=12)
    ax1.set_title('Soluciones de y\' + RBF(y) = sin(t)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Gráfica 2: Verificación de la ODE (residuos)
    ax2 = axes[0, 1]

    for idx, (name, data) in enumerate(solutions_dict.items()):
        if 'sol' in data and 'rbf' in data:
            sol = data['sol']
            rbf = data['rbf']

            if hasattr(sol, 'sol'):
                y_vals = sol.sol(t_plot)[0]
                # Calcular y' numéricamente
                y_prime = np.gradient(y_vals, t_plot)
                # Calcular residuo: y' + RBF(y) - sin(t)
                residual = y_prime + rbf(y_vals) - np.sin(t_plot)

                ax2.plot(t_plot, residual, label=name, linewidth=2,
                        color=colors[idx % len(colors)])

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('t', fontsize=12)
    ax2.set_ylabel('Residuo: y\' + RBF(y) - sin(t)', fontsize=12)
    ax2.set_title('Verificación de la ODE', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Gráfica 3: Función RBF
    ax3 = axes[1, 0]

    y_range = np.linspace(-2, 2, 200)
    for idx, (name, data) in enumerate(solutions_dict.items()):
        if 'rbf' in data:
            rbf = data['rbf']
            rbf_vals = rbf(y_range)
            ax3.plot(y_range, rbf_vals, label=f'RBF - {name}', linewidth=2,
                    color=colors[idx % len(colors)])

    ax3.set_xlabel('y', fontsize=12)
    ax3.set_ylabel('RBF(y)', fontsize=12)
    ax3.set_title('Función RBF(y)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Gráfica 4: Plano de fase y' vs y
    ax4 = axes[1, 1]

    for idx, (name, data) in enumerate(solutions_dict.items()):
        if 'sol' in data and 'rbf' in data:
            sol = data['sol']
            rbf = data['rbf']

            if hasattr(sol, 'sol'):
                y_vals = sol.sol(t_plot)[0]
                y_prime = np.gradient(y_vals, t_plot)

                ax4.plot(y_vals, y_prime, label=name, linewidth=2,
                        color=colors[idx % len(colors)], alpha=0.7)

    ax4.set_xlabel('y', fontsize=12)
    ax4.set_ylabel('y\'', fontsize=12)
    ax4.set_title('Plano de Fase', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/rodo/1Paper/ode_rbf_solutions.png', dpi=150, bbox_inches='tight')
    print("\n📊 Gráfica guardada: ode_rbf_solutions.png")
    plt.show()


def main():
    """Resolver la ODE con diferentes métodos"""

    print("\n" + "🔵"*35)
    print("RESOLVER ODE: y' + RBF(y) = sin(t)")
    print("🔵"*35)

    # Parámetros
    t_span = (0, 10)
    y0 = 0.5

    print(f"\nCondiciones:")
    print(f"  Ecuación: y' + RBF(y) = sin(t)")
    print(f"  Dominio: t ∈ [{t_span[0]}, {t_span[1]}]")
    print(f"  Condición inicial: y({t_span[0]}) = {y0}")

    # Definir RBF fija para Método 1
    print("\n" + "-"*70)
    print("Definiendo RBF para el término RBF(y)...")
    print("-"*70)

    # RBF simple: centros en [-1, 0, 1]
    centers = np.array([-1.0, 0.0, 1.0])
    sigma = 0.5
    weights = np.array([0.3, -0.5, 0.3, 0.1])  # + bias

    rbf1 = RBFNetwork(centers, sigma, weights)

    print(f"  Centros: {centers}")
    print(f"  Sigma: {sigma}")
    print(f"  Pesos: {weights[:-1]}, Bias: {weights[-1]}")

    # MÉTODO 1: Integración numérica
    sol1 = method1_numerical_integration(rbf1, t_span, y0)

    # MÉTODO 2: Colocación
    sol2_func, t_col = method2_collocation(
        n_centers_rbf=10,
        sigma_rbf=1.0,
        t_span=t_span,
        y0=y0,
        n_collocation=50
    )

    # MÉTODO 3: Optimización de parámetros
    rbf_structure = {
        'centers': np.array([-1.5, -0.5, 0.5, 1.5]),
        'sigma': 0.5
    }
    rbf3_opt, sol3 = method3_optimize_rbf_parameters(rbf_structure, t_span, y0, n_points=50)

    # Recopilar soluciones
    solutions = {
        'Método 1: Integración (RBF fija)': {
            'sol': sol1,
            'rbf': rbf1
        },
        'Método 2: Colocación': {
            'func': sol2_func
        },
        'Método 3: RBF optimizada': {
            'sol': sol3,
            'rbf': rbf3_opt
        }
    }

    # Visualizar
    visualize_solutions(solutions, t_span)

    print("\n" + "="*70)
    print("✓ Todos los métodos completados")
    print("="*70)


if __name__ == "__main__":
    main()
