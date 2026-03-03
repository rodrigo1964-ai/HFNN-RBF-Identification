"""
IDENTIFICACIÓN DE SISTEMA CON RBF

Problema: Tenemos datos experimentales (t_i, y_i) que sabemos satisfacen:
          y' + f(y) = sin(t)

          donde f(y) es DESCONOCIDA

Objetivo:
1. De los datos (t_i, y_i), inferir f(y)
2. Aproximar f(y) con una RBF
3. Validar la RBF resolviendo la ODE y comparando con datos

Enfoque:
- Calcular y'_i de los datos numéricamente
- Usar ecuación: f(y_i) = sin(t_i) - y'_i
- Entrenar RBF: f(y) ≈ RBF(y)
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

    def _gaussian_rbf(self, y, centers):
        """Función gaussiana"""
        y = np.array(y).reshape(-1, 1)
        centers = np.array(centers).reshape(-1, 1)
        distances = cdist(y, centers, 'euclidean')
        return np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

    def fit(self, y_data, f_data):
        """
        Entrenar RBF para aproximar f(y) desde datos

        Args:
            y_data: valores de y (variable independiente)
            f_data: valores de f(y) (variable dependiente)
        """
        y_data = np.array(y_data).flatten()
        f_data = np.array(f_data).flatten()

        # Colocar centros uniformemente en el rango de y_data
        y_min, y_max = y_data.min(), y_data.max()
        # Añadir margen
        margin = (y_max - y_min) * 0.1
        self.centers = np.linspace(y_min - margin, y_max + margin, self.n_centers)

        # Calcular matriz de activaciones
        phi = self._gaussian_rbf(y_data, self.centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])

        # Resolver mínimos cuadrados
        self.weights = np.linalg.lstsq(phi, f_data, rcond=None)[0]

        # Calcular error de aproximación
        f_pred = self.predict(y_data)
        mse = np.mean((f_pred - f_data) ** 2)
        r2 = 1 - np.sum((f_data - f_pred)**2) / np.sum((f_data - np.mean(f_data))**2)

        return mse, r2

    def predict(self, y):
        """Predecir f(y)"""
        y = np.array(y).flatten()
        phi = self._gaussian_rbf(y, self.centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        return phi @ self.weights

    def __call__(self, y):
        return self.predict(y)


def generate_experimental_data(t_span, y0, n_points=100, noise_level=0.0):
    """
    PASO 1: Generar datos experimentales "sintéticos"

    En la práctica, estos datos vendrían de experimentos reales.
    Aquí los simulamos resolviendo y' + y² = sin(t)
    """
    print("\n" + "="*70)
    print("PASO 1: Generar datos experimentales")
    print("="*70)

    # Resolver ODE "real" (que en la práctica no conocemos)
    def ode_real(t, y):
        return np.sin(t) - y**2  # f(y) = y² (desconocida para nosotros)

    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    sol = solve_ivp(
        ode_real,
        t_span,
        [y0],
        t_eval=t_eval,
        method='RK45',
        rtol=1e-10,
        atol=1e-12
    )

    t_data = sol.t
    y_data = sol.y[0]

    # Añadir ruido experimental (opcional)
    if noise_level > 0:
        y_data = y_data + np.random.normal(0, noise_level, len(y_data))

    print(f"  Puntos de datos generados: {len(t_data)}")
    print(f"  Rango temporal: t ∈ [{t_data[0]:.2f}, {t_data[-1]:.2f}]")
    print(f"  Rango de y: y ∈ [{y_data.min():.4f}, {y_data.max():.4f}]")
    print(f"  Nivel de ruido: {noise_level}")

    return t_data, y_data


def identify_function_from_data(t_data, y_data):
    """
    PASO 2: Identificar f(y) desde los datos

    De la ecuación: y' + f(y) = sin(t)
    Despejamos: f(y) = sin(t) - y'
    """
    print("\n" + "="*70)
    print("PASO 2: Identificar f(y) desde datos")
    print("="*70)

    # Calcular y' numéricamente usando diferencias finitas
    # Usamos gradiente centrado para mejor precisión
    y_prime = np.gradient(y_data, t_data)

    # De la ecuación: f(y) = sin(t) - y'
    f_data = np.sin(t_data) - y_prime

    print(f"  Derivada y' calculada numéricamente")
    print(f"  Rango de y': [{y_prime.min():.4f}, {y_prime.max():.4f}]")
    print(f"  f(y) inferida de la ecuación: f(y) = sin(t) - y'")
    print(f"  Rango de f(y): [{f_data.min():.4f}, {f_data.max():.4f}]")

    # Información sobre la calidad de la derivada numérica
    dt = np.diff(t_data)
    print(f"  Paso temporal promedio: {dt.mean():.4f}")
    print(f"  Paso temporal min/max: [{dt.min():.4f}, {dt.max():.4f}]")

    return y_prime, f_data


def train_rbf_from_identified_data(y_data, f_data, n_centers, sigma):
    """
    PASO 3: Entrenar RBF para aproximar f(y)

    Entrenamos: RBF(y) ≈ f(y)
    usando los pares (y_i, f_i) obtenidos de los datos
    """
    print("\n" + "="*70)
    print(f"PASO 3: Entrenar RBF con {n_centers} centros")
    print("="*70)

    rbf = RBFNetwork(n_centers=n_centers, sigma=sigma)
    mse, r2 = rbf.fit(y_data, f_data)

    print(f"  Centros RBF: {n_centers}")
    print(f"  Sigma: {sigma}")
    print(f"  MSE de aproximación: {mse:.6e}")
    print(f"  R² score: {r2:.6f}")

    # Mostrar estadísticas de los pesos
    print(f"  Pesos (sin bias): min={rbf.weights[:-1].min():.4f}, "
          f"max={rbf.weights[:-1].max():.4f}")
    print(f"  Bias: {rbf.weights[-1]:.4f}")

    return rbf


def validate_rbf_solution(rbf, t_span, y0, t_data_original, y_data_original):
    """
    PASO 4: Validar la RBF resolviendo la ODE

    Resolvemos: y' + RBF(y) = sin(t)
    y comparamos con los datos originales
    """
    print("\n" + "="*70)
    print("PASO 4: Validar RBF resolviendo ODE")
    print("="*70)

    def ode_with_rbf(t, y):
        """y' = sin(t) - RBF(y)"""
        return np.sin(t) - rbf(y)

    # Resolver ODE con RBF
    sol = solve_ivp(
        ode_with_rbf,
        t_span,
        [y0],
        dense_output=True,
        method='RK45',
        rtol=1e-10,
        atol=1e-12
    )

    # Evaluar en los mismos puntos que los datos originales
    y_pred = sol.sol(t_data_original)[0]

    # Calcular errores
    mse = np.mean((y_pred - y_data_original)**2)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(y_pred - y_data_original))
    r2 = 1 - np.sum((y_data_original - y_pred)**2) / \
         np.sum((y_data_original - np.mean(y_data_original))**2)

    print(f"  Estado: {sol.message}")
    print(f"  Evaluaciones: {sol.nfev}")
    print(f"\n  Comparación con datos originales:")
    print(f"    MSE:  {mse:.6e}")
    print(f"    RMSE: {rmse:.6e}")
    print(f"    Error máximo: {max_error:.6e}")
    print(f"    R² score: {r2:.6f}")

    return sol, y_pred, {'mse': mse, 'rmse': rmse, 'max_error': max_error, 'r2': r2}


def visualize_complete_analysis(t_data, y_data, y_prime, f_data, rbf_configs,
                                t_span, y0):
    """
    Visualización completa del análisis
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    colors = ['blue', 'red', 'green', 'purple', 'orange']

    # Gráfica 1: Datos experimentales y(t)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_data, y_data, 'ko-', markersize=4, linewidth=1.5,
             label='Datos experimentales', alpha=0.7)
    ax1.set_xlabel('t', fontsize=11)
    ax1.set_ylabel('y(t)', fontsize=11)
    ax1.set_title('Datos Experimentales', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Gráfica 2: Derivada calculada y'(t)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_data, y_prime, 'mo-', markersize=4, linewidth=1.5,
             label="y' (calculada)", alpha=0.7)
    ax2.set_xlabel('t', fontsize=11)
    ax2.set_ylabel("y'(t)", fontsize=11)
    ax2.set_title('Derivada Numérica', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Gráfica 3: f(y) identificada vs y
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(y_data, f_data, 'co', markersize=5, label='f(y) identificada',
             alpha=0.6)

    # Si conocemos la función real (solo para validación)
    y_range = np.linspace(y_data.min() - 0.1, y_data.max() + 0.1, 200)
    f_real = y_range ** 2
    ax3.plot(y_range, f_real, 'k--', linewidth=2, label='f(y) = y² (real)',
             alpha=0.7)

    ax3.set_xlabel('y', fontsize=11)
    ax3.set_ylabel('f(y)', fontsize=11)
    ax3.set_title('Función f(y) Identificada', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Gráfica 4: Aproximación de f(y) con diferentes RBFs
    ax4 = fig.add_subplot(gs[1, :])

    # Datos identificados
    ax4.scatter(y_data, f_data, c='gray', s=30, alpha=0.5,
                label='f(y) de datos', zorder=1)

    # Función real (referencia)
    ax4.plot(y_range, f_real, 'k-', linewidth=3, label='f(y) = y² (real)',
             alpha=0.7, zorder=10)

    # Aproximaciones RBF
    for idx, config in enumerate(rbf_configs):
        rbf = config['rbf']
        n_centers = config['n_centers']
        f_rbf = rbf(y_range)
        ax4.plot(y_range, f_rbf, '--', linewidth=2.5, color=colors[idx],
                label=f'RBF ({n_centers} centros)', alpha=0.8, zorder=5)

        # Mostrar centros
        ax4.scatter(rbf.centers, rbf(rbf.centers), marker='X', s=150,
                   color=colors[idx], edgecolors='black', linewidths=1.5,
                   zorder=6)

    ax4.set_xlabel('y', fontsize=12)
    ax4.set_ylabel('f(y)', fontsize=12)
    ax4.set_title('Aproximación de f(y) con RBF', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3)

    # Gráfica 5: Soluciones y(t) - Datos vs RBF
    ax5 = fig.add_subplot(gs[2, :])

    ax5.plot(t_data, y_data, 'ko', markersize=6, label='Datos experimentales',
             zorder=10, alpha=0.7)

    for idx, config in enumerate(rbf_configs):
        sol = config['sol']
        n_centers = config['n_centers']

        t_plot = np.linspace(t_span[0], t_span[1], 500)
        y_pred = sol.sol(t_plot)[0]

        ax5.plot(t_plot, y_pred, '-', linewidth=2.5, color=colors[idx],
                label=f'RBF ({n_centers} centros)', alpha=0.7)

    ax5.set_xlabel('t', fontsize=12)
    ax5.set_ylabel('y(t)', fontsize=12)
    ax5.set_title("Validación: y(t) predicha vs datos experimentales",
                 fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10, loc='best')
    ax5.grid(True, alpha=0.3)

    # Gráfica 6: Error de predicción
    ax6 = fig.add_subplot(gs[3, 0])

    for idx, config in enumerate(rbf_configs):
        y_pred = config['y_pred']
        n_centers = config['n_centers']
        error = np.abs(y_pred - y_data)

        ax6.semilogy(t_data, error + 1e-12, 'o-', linewidth=2, markersize=4,
                    color=colors[idx], label=f'{n_centers} centros', alpha=0.7)

    ax6.set_xlabel('t', fontsize=11)
    ax6.set_ylabel('|y_pred - y_data|', fontsize=11)
    ax6.set_title('Error de Predicción', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, which='both')

    # Gráfica 7: Error de aproximación de f(y)
    ax7 = fig.add_subplot(gs[3, 1])

    for idx, config in enumerate(rbf_configs):
        rbf = config['rbf']
        n_centers = config['n_centers']
        f_rbf = rbf(y_data)
        error_f = np.abs(f_rbf - f_data)

        ax7.semilogy(y_data, error_f + 1e-12, 'o', markersize=5,
                    color=colors[idx], label=f'{n_centers} centros', alpha=0.7)

    ax7.set_xlabel('y', fontsize=11)
    ax7.set_ylabel('|RBF(y) - f_datos(y)|', fontsize=11)
    ax7.set_title('Error de Aproximación de f(y)', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, which='both')

    # Gráfica 8: Tabla resumen
    ax8 = fig.add_subplot(gs[3, 2])
    ax8.axis('off')

    # Crear tabla de resultados
    table_data = []
    headers = ['Centros', 'MSE f(y)', 'R² f(y)', 'RMSE y(t)', 'R² y(t)']

    for config in rbf_configs:
        n_centers = config['n_centers']
        mse_f = config['mse_f']
        r2_f = config['r2_f']
        errors = config['errors']

        table_data.append([
            f"{n_centers}",
            f"{mse_f:.2e}",
            f"{r2_f:.4f}",
            f"{errors['rmse']:.2e}",
            f"{errors['r2']:.4f}"
        ])

    table = ax8.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.2, 0.18, 0.2, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Estilo de la tabla
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax8.set_title('Resumen de Resultados', fontsize=12, fontweight='bold',
                 pad=20)

    plt.savefig('/home/rodo/1Paper/ode_rbf_identification_complete.png',
                dpi=150, bbox_inches='tight')
    print("\n📊 Gráfica guardada: ode_rbf_identification_complete.png")
    plt.close()


def main():
    """Programa principal - Identificación completa"""

    print("\n" + "🔵"*35)
    print("IDENTIFICACIÓN DE SISTEMA CON RBF")
    print("Ecuación: y' + f(y) = sin(t) con f desconocida")
    print("🔵"*35)

    # Configuración
    t_span = (0, 5)
    y0 = 0.5
    n_data_points = 50
    noise_level = 0.0  # 0.01 para añadir ruido

    print(f"\n📋 CONFIGURACIÓN:")
    print(f"  Dominio temporal: t ∈ [{t_span[0]}, {t_span[1]}]")
    print(f"  Condición inicial: y(0) = {y0}")
    print(f"  Puntos de datos: {n_data_points}")
    print(f"  Nivel de ruido: {noise_level}")

    # PASO 1: Generar/obtener datos experimentales
    t_data, y_data = generate_experimental_data(t_span, y0, n_data_points, noise_level)

    # PASO 2: Identificar f(y) desde los datos
    y_prime, f_data = identify_function_from_data(t_data, y_data)

    # PASO 3: Entrenar RBFs con diferentes configuraciones
    rbf_configs = []

    for n_centers in [5, 10, 15]:
        rbf = train_rbf_from_identified_data(y_data, f_data, n_centers, sigma=0.3)

        # PASO 4: Validar
        sol, y_pred, errors = validate_rbf_solution(rbf, t_span, y0, t_data, y_data)

        # Guardar configuración
        rbf_configs.append({
            'n_centers': n_centers,
            'rbf': rbf,
            'sol': sol,
            'y_pred': y_pred,
            'errors': errors,
            'mse_f': np.mean((rbf(y_data) - f_data)**2),
            'r2_f': 1 - np.sum((f_data - rbf(y_data))**2) / \
                    np.sum((f_data - np.mean(f_data))**2)
        })

    # Visualización completa
    print("\n" + "="*70)
    print("PASO 5: Visualización completa")
    print("="*70)

    visualize_complete_analysis(t_data, y_data, y_prime, f_data,
                               rbf_configs, t_span, y0)

    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"\n{'Config':<12} {'MSE f(y)':<12} {'R² f(y)':<10} {'RMSE y(t)':<12} {'R² y(t)':<10}")
    print("-"*70)

    for config in rbf_configs:
        print(f"{config['n_centers']:>3} centros   "
              f"{config['mse_f']:<12.4e} "
              f"{config['r2_f']:<10.6f} "
              f"{config['errors']['rmse']:<12.4e} "
              f"{config['errors']['r2']:<10.6f}")

    print("="*70)
    print("\n✓ Identificación completada exitosamente!")
    print("\n💡 CONCLUSIÓN:")
    print("  - De datos experimentales (t, y), identificamos f(y)")
    print("  - RBF aproxima bien f(y) desde los datos")
    print("  - La ODE con RBF reproduce los datos originales con alta precisión")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
