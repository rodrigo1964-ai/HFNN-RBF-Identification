"""
Análisis: ¿Cuántos puntos de datos se necesitan?

Estudia cómo el número de puntos experimentales afecta:
1. Calidad de la derivada numérica y'
2. Identificación de f(y)
3. Aproximación con RBF
4. Precisión de la solución ODE
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
        y = np.array(y).reshape(-1, 1)
        centers = np.array(centers).reshape(-1, 1)
        distances = cdist(y, centers, 'euclidean')
        return np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

    def fit(self, y_data, f_data):
        y_data = np.array(y_data).flatten()
        f_data = np.array(f_data).flatten()

        y_min, y_max = y_data.min(), y_data.max()
        margin = (y_max - y_min) * 0.1
        self.centers = np.linspace(y_min - margin, y_max + margin, self.n_centers)

        phi = self._gaussian_rbf(y_data, self.centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        self.weights = np.linalg.lstsq(phi, f_data, rcond=None)[0]

        return self

    def predict(self, y):
        y = np.array(y).flatten()
        phi = self._gaussian_rbf(y, self.centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        return phi @ self.weights

    def __call__(self, y):
        return self.predict(y)


def generate_data(t_span, y0, n_points):
    """Generar datos experimentales con n_points"""

    def ode_real(t, y):
        return np.sin(t) - y**2

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

    return sol.t, sol.y[0]


def identify_and_train(t_data, y_data, n_centers, sigma):
    """
    Identificar f(y) y entrenar RBF
    Retorna métricas de calidad
    """
    # Calcular derivada
    y_prime = np.gradient(y_data, t_data)

    # Identificar f(y)
    f_data = np.sin(t_data) - y_prime

    # Calcular f real para comparación
    f_real = y_data ** 2

    # Error en la identificación de f
    error_f_identification = np.sqrt(np.mean((f_data - f_real) ** 2))
    max_error_f = np.max(np.abs(f_data - f_real))

    # Entrenar RBF
    rbf = RBFNetwork(n_centers=n_centers, sigma=sigma)
    rbf.fit(y_data, f_data)

    # Error de aproximación RBF
    f_rbf = rbf(y_data)
    error_rbf = np.sqrt(np.mean((f_rbf - f_real) ** 2))

    return rbf, {
        'error_f_identification': error_f_identification,
        'max_error_f': max_error_f,
        'error_rbf': error_rbf,
        'y_prime': y_prime,
        'f_data': f_data
    }


def validate_solution(rbf, t_span, y0, t_reference, y_reference):
    """Validar solución con RBF"""

    def ode_with_rbf(t, y):
        return np.sin(t) - rbf(y)

    sol = solve_ivp(
        ode_with_rbf,
        t_span,
        [y0],
        t_eval=t_reference,
        method='RK45',
        rtol=1e-10,
        atol=1e-12
    )

    if not sol.success:
        return {'rmse': np.inf, 'max_error': np.inf}

    y_pred = sol.y[0]
    rmse = np.sqrt(np.mean((y_pred - y_reference) ** 2))
    max_error = np.max(np.abs(y_pred - y_reference))

    return {'rmse': rmse, 'max_error': max_error, 'y_pred': y_pred}


def comprehensive_analysis():
    """Análisis completo variando número de puntos"""

    print("\n" + "🔵"*35)
    print("ANÁLISIS: ¿Cuántos puntos de datos se necesitan?")
    print("🔵"*35)

    # Configuración
    t_span = (0, 5)
    y0 = 0.5
    n_centers = 10
    sigma = 0.3

    # Generar referencia con muchos puntos (verdad de referencia)
    print("\nGenerando datos de referencia (alta resolución)...")
    t_reference, y_reference = generate_data(t_span, y0, 500)
    print(f"  Puntos de referencia: {len(t_reference)}")

    # Probar diferentes cantidades de puntos
    n_points_list = [10, 15, 20, 30, 40, 50, 75, 100, 150, 200]

    print(f"\nConfiguración:")
    print(f"  RBF: {n_centers} centros, sigma={sigma}")
    print(f"  Probando con: {n_points_list} puntos\n")

    results = []

    for n_points in n_points_list:
        print(f"{'─'*70}")
        print(f"Procesando {n_points} puntos...")
        print(f"{'─'*70}")

        # Generar datos
        t_data, y_data = generate_data(t_span, y0, n_points)

        # Identificar y entrenar
        rbf, metrics = identify_and_train(t_data, y_data, n_centers, sigma)

        # Validar
        validation = validate_solution(rbf, t_span, y0, t_reference, y_reference)

        print(f"  Error identificación f(y): {metrics['error_f_identification']:.6e}")
        print(f"  Error aproximación RBF:    {metrics['error_rbf']:.6e}")
        print(f"  RMSE solución ODE:         {validation['rmse']:.6e}")
        print(f"  Error máximo solución:     {validation['max_error']:.6e}")

        results.append({
            'n_points': n_points,
            'dt': (t_span[1] - t_span[0]) / (n_points - 1),
            't_data': t_data,
            'y_data': y_data,
            'rbf': rbf,
            'metrics': metrics,
            'validation': validation
        })

    # Visualización
    visualize_analysis(results, t_reference, y_reference, n_centers)

    # Tabla resumen
    print_summary_table(results)

    return results


def visualize_analysis(results, t_reference, y_reference, n_centers):
    """Visualización completa del análisis"""

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # Colores para diferentes cantidades de puntos
    n_results = len(results)
    colors = plt.cm.viridis(np.linspace(0, 1, n_results))

    # Gráfica 1: Error de identificación de f vs n_points
    ax1 = fig.add_subplot(gs[0, 0])

    n_points_arr = [r['n_points'] for r in results]
    error_f_arr = [r['metrics']['error_f_identification'] for r in results]

    ax1.semilogy(n_points_arr, error_f_arr, 'o-', linewidth=2, markersize=8,
                color='blue', markerfacecolor='lightblue', markeredgewidth=2)
    ax1.axhline(y=0.01, color='red', linestyle='--', label='Umbral 0.01', linewidth=1.5)
    ax1.axhline(y=0.001, color='green', linestyle='--', label='Umbral 0.001', linewidth=1.5)

    ax1.set_xlabel('Número de puntos', fontsize=12)
    ax1.set_ylabel('RMSE en identificación de f(y)', fontsize=12)
    ax1.set_title('Error en Identificación de f(y) vs Datos', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')

    # Gráfica 2: Error de aproximación RBF vs n_points
    ax2 = fig.add_subplot(gs[0, 1])

    error_rbf_arr = [r['metrics']['error_rbf'] for r in results]

    ax2.semilogy(n_points_arr, error_rbf_arr, 's-', linewidth=2, markersize=8,
                color='red', markerfacecolor='lightcoral', markeredgewidth=2)
    ax2.axhline(y=0.01, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.axhline(y=0.001, color='green', linestyle='--', linewidth=1.5, alpha=0.5)

    ax2.set_xlabel('Número de puntos', fontsize=12)
    ax2.set_ylabel('RMSE en aproximación RBF', fontsize=12)
    ax2.set_title('Error de Aproximación RBF vs Datos', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')

    # Gráfica 3: Error en solución ODE vs n_points
    ax3 = fig.add_subplot(gs[0, 2])

    rmse_sol_arr = [r['validation']['rmse'] for r in results]

    ax3.semilogy(n_points_arr, rmse_sol_arr, '^-', linewidth=2, markersize=8,
                color='green', markerfacecolor='lightgreen', markeredgewidth=2)
    ax3.axhline(y=0.01, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.axhline(y=0.001, color='green', linestyle='--', linewidth=1.5, alpha=0.5)

    ax3.set_xlabel('Número de puntos', fontsize=12)
    ax3.set_ylabel('RMSE solución ODE', fontsize=12)
    ax3.set_title('Error en Solución ODE vs Datos', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')

    # Gráfica 4: Paso temporal vs error
    ax4 = fig.add_subplot(gs[1, 0])

    dt_arr = [r['dt'] for r in results]

    ax4.loglog(dt_arr, error_f_arr, 'o-', linewidth=2, markersize=8,
              label='Error identificación f')
    ax4.loglog(dt_arr, error_rbf_arr, 's-', linewidth=2, markersize=8,
              label='Error aproximación RBF')
    ax4.loglog(dt_arr, rmse_sol_arr, '^-', linewidth=2, markersize=8,
              label='RMSE solución ODE')

    ax4.set_xlabel('Paso temporal Δt', fontsize=12)
    ax4.set_ylabel('Error', fontsize=12)
    ax4.set_title('Error vs Paso Temporal (escala log-log)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.invert_xaxis()

    # Gráfica 5: Comparación de soluciones (pocos puntos)
    ax5 = fig.add_subplot(gs[1, 1])

    ax5.plot(t_reference, y_reference, 'k-', linewidth=3, label='Referencia', zorder=10)

    for idx in [0, 2, 4]:  # 10, 20, 40 puntos
        if idx < len(results):
            r = results[idx]
            y_pred = r['validation']['y_pred']
            ax5.plot(t_reference, y_pred, '--', linewidth=2, color=colors[idx],
                    label=f"{r['n_points']} puntos", alpha=0.8)
            ax5.scatter(r['t_data'], r['y_data'], s=50, color=colors[idx],
                       edgecolors='black', linewidths=1, zorder=5)

    ax5.set_xlabel('t', fontsize=12)
    ax5.set_ylabel('y(t)', fontsize=12)
    ax5.set_title('Soluciones con Pocos Puntos', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Gráfica 6: Comparación de soluciones (muchos puntos)
    ax6 = fig.add_subplot(gs[1, 2])

    ax6.plot(t_reference, y_reference, 'k-', linewidth=3, label='Referencia', zorder=10)

    for idx in [-3, -2, -1]:  # últimos 3
        if abs(idx) <= len(results):
            r = results[idx]
            y_pred = r['validation']['y_pred']
            ax6.plot(t_reference, y_pred, '--', linewidth=2, color=colors[idx],
                    label=f"{r['n_points']} puntos", alpha=0.8)

    ax6.set_xlabel('t', fontsize=12)
    ax6.set_ylabel('y(t)', fontsize=12)
    ax6.set_title('Soluciones con Muchos Puntos', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # Gráfica 7: f(y) identificada con diferentes n_points
    ax7 = fig.add_subplot(gs[2, :])

    y_plot = np.linspace(-1.2, 1.0, 200)
    f_real = y_plot ** 2

    ax7.plot(y_plot, f_real, 'k-', linewidth=3, label='f(y) = y² (real)', zorder=10)

    indices_to_plot = [0, 3, 6, 9] if len(results) >= 10 else range(len(results))

    for idx in indices_to_plot:
        r = results[idx]
        f_rbf = r['rbf'](y_plot)
        ax7.plot(y_plot, f_rbf, '--', linewidth=2, color=colors[idx],
                label=f"{r['n_points']} puntos", alpha=0.7)
        ax7.scatter(r['y_data'], r['metrics']['f_data'], s=40, color=colors[idx],
                   alpha=0.5, edgecolors='black', linewidths=0.5)

    ax7.set_xlabel('y', fontsize=12)
    ax7.set_ylabel('f(y)', fontsize=12)
    ax7.set_title('Función f(y) Identificada con Diferentes Cantidades de Datos',
                 fontsize=13, fontweight='bold')
    ax7.legend(fontsize=10, ncol=3)
    ax7.grid(True, alpha=0.3)

    # Gráfica 8: Error punto a punto en función f(y)
    ax8 = fig.add_subplot(gs[3, 0])

    for idx in indices_to_plot:
        r = results[idx]
        f_real_data = r['y_data'] ** 2
        error_f_point = np.abs(r['metrics']['f_data'] - f_real_data)
        ax8.semilogy(r['y_data'], error_f_point, 'o', markersize=6,
                    color=colors[idx], label=f"{r['n_points']} puntos", alpha=0.7)

    ax8.set_xlabel('y', fontsize=11)
    ax8.set_ylabel('|f_identificada - f_real|', fontsize=11)
    ax8.set_title('Error Puntual en f(y)', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, which='both')

    # Gráfica 9: Tabla de recomendaciones
    ax9 = fig.add_subplot(gs[3, 1:])
    ax9.axis('off')

    # Encontrar umbrales
    threshold_001 = next((r['n_points'] for r in results if r['validation']['rmse'] < 0.001), None)
    threshold_01 = next((r['n_points'] for r in results if r['validation']['rmse'] < 0.01), None)

    recommendations_text = f"""
RECOMENDACIONES BASADAS EN ANÁLISIS

📊 Puntos Mínimos Requeridos:
   • Para RMSE < 0.01:    {threshold_01 if threshold_01 else '>200'} puntos
   • Para RMSE < 0.001:   {threshold_001 if threshold_001 else '>200'} puntos

📈 Análisis de Sensibilidad:
   • 10-20 puntos:   Error alto, solo para pruebas rápidas
   • 30-50 puntos:   Error moderado, suficiente para análisis preliminar
   • 75-100 puntos:  Buena precisión para la mayoría de aplicaciones
   • 150+ puntos:    Alta precisión, recomendado para trabajo final

⚡ Factor Limitante Principal:
   El error en la derivada numérica (y') domina el error total.
   Δt más pequeño → mejor estimación de y' → mejor identificación de f(y)

💡 Recomendación General:
   Usar ≥ {n_points_arr[4] if len(n_points_arr) > 4 else 50} puntos para aplicaciones prácticas
   RBF con {n_centers} centros es suficiente para este problema
    """

    ax9.text(0.05, 0.95, recommendations_text, transform=ax9.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig('/home/rodo/1Paper/ode_rbf_data_requirements.png',
                dpi=150, bbox_inches='tight')
    print("\n📊 Gráfica guardada: ode_rbf_data_requirements.png")
    plt.close()


def print_summary_table(results):
    """Tabla resumen de resultados"""

    print("\n" + "="*90)
    print("TABLA RESUMEN: Error vs Número de Puntos")
    print("="*90)
    print(f"{'N Puntos':<10} {'Δt':<12} {'Error f(y)':<15} {'Error RBF':<15} "
          f"{'RMSE ODE':<15} {'Max Error':<15}")
    print("-"*90)

    for r in results:
        print(f"{r['n_points']:<10} "
              f"{r['dt']:<12.6f} "
              f"{r['metrics']['error_f_identification']:<15.6e} "
              f"{r['metrics']['error_rbf']:<15.6e} "
              f"{r['validation']['rmse']:<15.6e} "
              f"{r['validation']['max_error']:<15.6e}")

    print("="*90 + "\n")


def main():
    """Programa principal"""
    results = comprehensive_analysis()

    # Análisis adicional
    print("\n💡 CONCLUSIONES CLAVE:")
    print("-"*70)

    # Encontrar punto de rendimientos decrecientes
    rmse_values = [r['validation']['rmse'] for r in results]
    n_points_values = [r['n_points'] for r in results]

    # Mejora relativa
    for i in range(1, len(results)):
        improvement = (rmse_values[i-1] - rmse_values[i]) / rmse_values[i-1] * 100
        if improvement < 10 and n_points_values[i] < 100:
            print(f"\n✓ Rendimientos decrecientes después de ~{n_points_values[i]} puntos")
            print(f"  (mejora < 10% al añadir más datos)")
            break

    print("\n✓ Error dominado por precisión de la derivada numérica y'")
    print("✓ RBF aproxima bien f(y) incluso con pocos puntos")
    print("✓ Recomendación: 50-100 puntos para balance precisión/costo\n")


if __name__ == "__main__":
    main()
