"""
Resolver ODE: y' + RBF(y) = sin(t)

donde RBF(y) ≈ y² (función desconocida a aproximar)

Ecuación real: y' + y² = sin(t)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist


class RBFNetwork:
    """Red de Base Radial"""

    def __init__(self, n_centers, sigma=1.0):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _gaussian_rbf(self, X, centers):
        """Función gaussiana"""
        distances = cdist(X.reshape(-1, 1), centers.reshape(-1, 1), 'euclidean')
        return np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

    def fit(self, y_train, z_train):
        """
        Entrenar RBF para aproximar z = f(y)

        Args:
            y_train: valores de entrada (y)
            z_train: valores objetivo (z = y²)
        """
        y_train = np.array(y_train).flatten()
        z_train = np.array(z_train).flatten()

        # Colocar centros uniformemente en el rango de y_train
        y_min, y_max = y_train.min(), y_train.max()
        self.centers = np.linspace(y_min, y_max, self.n_centers)

        # Calcular matriz de activaciones
        phi = self._gaussian_rbf(y_train, self.centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])  # añadir bias

        # Resolver mínimos cuadrados
        self.weights = np.linalg.lstsq(phi, z_train, rcond=None)[0]

        # Calcular error de aproximación
        z_pred = self.predict(y_train)
        mse = np.mean((z_pred - z_train) ** 2)

        return mse

    def predict(self, y):
        """Predecir z = RBF(y)"""
        y = np.array(y).flatten()
        phi = self._gaussian_rbf(y, self.centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        return phi @ self.weights

    def __call__(self, y):
        """Permitir usar rbf(y)"""
        return self.predict(y)


def train_rbf_for_y_squared(n_centers, sigma, y_range=(-2, 2), n_samples=100):
    """
    Entrenar RBF para aproximar z = y²

    Args:
        n_centers: número de centros RBF
        sigma: parámetro de anchura
        y_range: rango de y para entrenamiento
        n_samples: número de muestras de entrenamiento
    """
    print(f"\n{'─'*70}")
    print(f"Entrenando RBF con {n_centers} centros para aproximar z = y²")
    print(f"{'─'*70}")

    # Generar datos de entrenamiento
    y_train = np.linspace(y_range[0], y_range[1], n_samples)
    z_train = y_train ** 2  # función objetivo: z = y²

    # Crear y entrenar RBF
    rbf = RBFNetwork(n_centers=n_centers, sigma=sigma)
    mse = rbf.fit(y_train, z_train)

    print(f"  Rango de entrenamiento: y ∈ [{y_range[0]}, {y_range[1]}]")
    print(f"  Muestras de entrenamiento: {n_samples}")
    print(f"  Sigma: {sigma}")
    print(f"  MSE de aproximación: {mse:.6e}")

    # Calcular error máximo
    z_pred = rbf.predict(y_train)
    max_error = np.max(np.abs(z_pred - z_train))
    print(f"  Error máximo: {max_error:.6e}")

    return rbf, y_train, z_train


def solve_ode_with_rbf(rbf, t_span, y0, label="RBF"):
    """
    Resolver y' + RBF(y) = sin(t)

    Args:
        rbf: red RBF entrenada
        t_span: intervalo de tiempo (t_min, t_max)
        y0: condición inicial
    """
    def ode_system(t, y):
        """y' = sin(t) - RBF(y)"""
        return np.sin(t) - rbf(y)

    sol = solve_ivp(
        ode_system,
        t_span,
        [y0],
        method='RK45',
        dense_output=True,
        max_step=0.01,
        rtol=1e-8,
        atol=1e-10
    )

    return sol


def solve_ode_exact(t_span, y0):
    """
    Resolver y' + y² = sin(t) (solución exacta)
    """
    def ode_system(t, y):
        """y' = sin(t) - y²"""
        return np.sin(t) - y ** 2

    sol = solve_ivp(
        ode_system,
        t_span,
        [y0],
        method='RK45',
        dense_output=True,
        max_step=0.01,
        rtol=1e-8,
        atol=1e-10
    )

    return sol


def visualize_results(rbf_configs, sol_exact, t_span, y0):
    """
    Visualizar resultados de diferentes configuraciones RBF vs solución exacta
    """
    t_plot = np.linspace(t_span[0], t_span[1], 500)
    y_exact = sol_exact.sol(t_plot)[0]

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    colors = ['blue', 'red', 'green', 'purple', 'orange']

    # Gráfica 1: Aproximación de y² por diferentes RBFs
    ax1 = fig.add_subplot(gs[0, 0])
    y_range = np.linspace(-2, 2, 200)
    y_squared = y_range ** 2

    ax1.plot(y_range, y_squared, 'k-', linewidth=3, label='y² (exacta)', zorder=10)

    for idx, config in enumerate(rbf_configs):
        rbf = config['rbf']
        n_centers = config['n_centers']
        rbf_approx = rbf(y_range)
        ax1.plot(y_range, rbf_approx, '--', linewidth=2,
                color=colors[idx], label=f'RBF ({n_centers} centros)', alpha=0.8)

    ax1.set_xlabel('y', fontsize=11)
    ax1.set_ylabel('z', fontsize=11)
    ax1.set_title('Aproximación de z = y² con RBF', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 4)

    # Gráfica 2: Error de aproximación
    ax2 = fig.add_subplot(gs[0, 1])

    for idx, config in enumerate(rbf_configs):
        rbf = config['rbf']
        n_centers = config['n_centers']
        error = np.abs(rbf(y_range) - y_squared)
        ax2.semilogy(y_range, error + 1e-12, linewidth=2,
                    color=colors[idx], label=f'{n_centers} centros')

    ax2.set_xlabel('y', fontsize=11)
    ax2.set_ylabel('|RBF(y) - y²|', fontsize=11)
    ax2.set_title('Error de Aproximación (escala log)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    # Gráfica 3: Centros y pesos de RBF
    ax3 = fig.add_subplot(gs[0, 2])

    for idx, config in enumerate(rbf_configs):
        rbf = config['rbf']
        n_centers = config['n_centers']
        ax3.scatter(rbf.centers, rbf.weights[:-1], s=100,
                   color=colors[idx], marker='o', edgecolors='black',
                   linewidth=1.5, label=f'{n_centers} centros', zorder=5, alpha=0.7)

    ax3.set_xlabel('Posición del centro', fontsize=11)
    ax3.set_ylabel('Peso', fontsize=11)
    ax3.set_title('Centros y Pesos de RBF', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Gráfica 4: Solución y(t) - Todas las RBF vs Exacta
    ax4 = fig.add_subplot(gs[1, :])

    ax4.plot(t_plot, y_exact, 'k-', linewidth=3, label='Exacta: y\' + y² = sin(t)', zorder=10)

    for idx, config in enumerate(rbf_configs):
        sol = config['sol']
        n_centers = config['n_centers']
        y_rbf = sol.sol(t_plot)[0]
        ax4.plot(t_plot, y_rbf, '--', linewidth=2,
                color=colors[idx], label=f'RBF ({n_centers} centros)', alpha=0.8)

    ax4.set_xlabel('t', fontsize=12)
    ax4.set_ylabel('y(t)', fontsize=12)
    ax4.set_title('Soluciones: y\' + RBF(y) = sin(t) vs y\' + y² = sin(t)',
                 fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3)

    # Gráfica 5: Error de la solución
    ax5 = fig.add_subplot(gs[2, 0])

    for idx, config in enumerate(rbf_configs):
        sol = config['sol']
        n_centers = config['n_centers']
        y_rbf = sol.sol(t_plot)[0]
        error = np.abs(y_rbf - y_exact)
        ax5.semilogy(t_plot, error + 1e-12, linewidth=2,
                    color=colors[idx], label=f'{n_centers} centros')

    ax5.set_xlabel('t', fontsize=11)
    ax5.set_ylabel('|y_RBF(t) - y_exacta(t)|', fontsize=11)
    ax5.set_title('Error de la Solución (escala log)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, which='both')

    # Gráfica 6: Verificación de la ODE (residuo)
    ax6 = fig.add_subplot(gs[2, 1])

    # Para la solución exacta
    y_prime_exact = np.gradient(y_exact, t_plot)
    residual_exact = y_prime_exact + y_exact**2 - np.sin(t_plot)
    ax6.plot(t_plot, residual_exact, 'k-', linewidth=2, label='Exacta', alpha=0.7)

    for idx, config in enumerate(rbf_configs):
        sol = config['sol']
        rbf = config['rbf']
        n_centers = config['n_centers']
        y_rbf = sol.sol(t_plot)[0]
        y_prime_rbf = np.gradient(y_rbf, t_plot)
        residual = y_prime_rbf + rbf(y_rbf) - np.sin(t_plot)
        ax6.plot(t_plot, residual, '--', linewidth=2,
                color=colors[idx], label=f'RBF ({n_centers} centros)', alpha=0.7)

    ax6.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax6.set_xlabel('t', fontsize=11)
    ax6.set_ylabel('Residuo: y\' + f(y) - sin(t)', fontsize=11)
    ax6.set_title('Verificación de la ODE', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # Gráfica 7: Plano de fase
    ax7 = fig.add_subplot(gs[2, 2])

    y_prime_exact_plot = np.gradient(y_exact, t_plot)
    ax7.plot(y_exact, y_prime_exact_plot, 'k-', linewidth=3,
            label='Exacta', zorder=10)

    for idx, config in enumerate(rbf_configs):
        sol = config['sol']
        n_centers = config['n_centers']
        y_rbf = sol.sol(t_plot)[0]
        y_prime_rbf = np.gradient(y_rbf, t_plot)
        ax7.plot(y_rbf, y_prime_rbf, '--', linewidth=2,
                color=colors[idx], label=f'RBF ({n_centers})', alpha=0.7)

    ax7.scatter([y0], [np.sin(0) - y0**2], s=200, c='red',
               marker='*', edgecolors='black', linewidths=2,
               label='C.I.', zorder=15)

    ax7.set_xlabel('y', fontsize=11)
    ax7.set_ylabel('y\'', fontsize=11)
    ax7.set_title('Plano de Fase', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    plt.savefig('/home/rodo/1Paper/ode_rbf_approximation.png', dpi=150, bbox_inches='tight')
    print("\n📊 Gráfica guardada: ode_rbf_approximation.png")

    plt.show()


def main():
    """Programa principal"""

    print("\n" + "🔵"*35)
    print("RESOLVER ODE: y' + RBF(y) = sin(t)")
    print("donde RBF(y) ≈ y² (función desconocida)")
    print("🔵"*35)

    # Parámetros de la ODE
    t_span = (0, 5)  # Dominio más corto para evitar singularidad
    y0 = 0.5

    print(f"\n📋 CONFIGURACIÓN:")
    print(f"{'─'*70}")
    print(f"  Ecuación: y' + RBF(y) = sin(t)")
    print(f"  Función objetivo: RBF(y) ≈ y²")
    print(f"  Ecuación real: y' + y² = sin(t)")
    print(f"  Dominio temporal: t ∈ [{t_span[0]}, {t_span[1]}]")
    print(f"  Condición inicial: y({t_span[0]}) = {y0}")

    # Paso 1: Resolver ecuación exacta
    print(f"\n{'='*70}")
    print("PASO 1: Resolver ecuación exacta y' + y² = sin(t)")
    print(f"{'='*70}")

    sol_exact = solve_ode_exact(t_span, y0)
    print(f"  Estado: {sol_exact.message}")
    print(f"  Evaluaciones de función: {sol_exact.nfev}")

    # Paso 2: Entrenar RBFs con diferentes configuraciones
    print(f"\n{'='*70}")
    print("PASO 2: Entrenar RBFs para aproximar y²")
    print(f"{'='*70}")

    rbf_configs = []

    # Configuración 1: Pocos centros
    rbf1, y_train1, z_train1 = train_rbf_for_y_squared(
        n_centers=3, sigma=0.5, y_range=(-2, 2), n_samples=100
    )
    rbf_configs.append({'n_centers': 3, 'rbf': rbf1})

    # Configuración 2: Centros moderados
    rbf2, y_train2, z_train2 = train_rbf_for_y_squared(
        n_centers=8, sigma=0.5, y_range=(-2, 2), n_samples=100
    )
    rbf_configs.append({'n_centers': 8, 'rbf': rbf2})

    # Configuración 3: Muchos centros
    rbf3, y_train3, z_train3 = train_rbf_for_y_squared(
        n_centers=15, sigma=0.5, y_range=(-2, 2), n_samples=100
    )
    rbf_configs.append({'n_centers': 15, 'rbf': rbf3})

    # Paso 3: Resolver ODE con cada RBF
    print(f"\n{'='*70}")
    print("PASO 3: Resolver y' + RBF(y) = sin(t) con cada RBF")
    print(f"{'='*70}")

    for config in rbf_configs:
        n_centers = config['n_centers']
        rbf = config['rbf']

        print(f"\n  Resolviendo con RBF de {n_centers} centros...")
        sol = solve_ode_with_rbf(rbf, t_span, y0)
        config['sol'] = sol

        print(f"    Estado: {sol.message}")
        print(f"    Evaluaciones: {sol.nfev}")

    # Paso 4: Calcular errores
    print(f"\n{'='*70}")
    print("PASO 4: Análisis de errores")
    print(f"{'='*70}")

    t_eval = np.linspace(t_span[0], t_span[1], 100)
    y_exact_eval = sol_exact.sol(t_eval)[0]

    print(f"\n{'Configuración':<25} {'MSE (aprox y²)':>15} {'MSE (solución)':>15} {'Error máx':>15}")
    print("─"*70)

    for config in rbf_configs:
        n_centers = config['n_centers']
        rbf = config['rbf']
        sol = config['sol']

        # Error de aproximación de y²
        y_test = np.linspace(-2, 2, 100)
        mse_approx = np.mean((rbf(y_test) - y_test**2) ** 2)

        # Error de la solución
        y_rbf_eval = sol.sol(t_eval)[0]
        mse_sol = np.mean((y_rbf_eval - y_exact_eval) ** 2)
        max_error = np.max(np.abs(y_rbf_eval - y_exact_eval))

        print(f"{n_centers:>3} centros{'':<16} {mse_approx:>15.6e} {mse_sol:>15.6e} {max_error:>15.6e}")

    # Visualizar
    print(f"\n{'='*70}")
    print("PASO 5: Visualización")
    print(f"{'='*70}")

    visualize_results(rbf_configs, sol_exact, t_span, y0)

    print(f"\n{'='*70}")
    print("✓ Análisis completado exitosamente")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
