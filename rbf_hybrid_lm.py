"""
RBF Híbrida: K-means para centros + Levenberg-Marquardt para pesos
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


class RBFHybridLM:
    """RBF con K-means para centros y L-M para optimizar pesos"""

    def __init__(self, n_centers, sigma=1.0, refine_centers=False):
        """
        Args:
            n_centers: número de centros
            sigma: ancho de gaussianas
            refine_centers: si True, L-M también ajusta ligeramente los centros
        """
        self.n_centers = n_centers
        self.sigma = sigma
        self.refine_centers = refine_centers
        self.centers = None
        self.weights = None
        self.n_features = None

    def _gaussian_rbf(self, X, centers):
        """Función de base radial gaussiana"""
        distances = cdist(X, centers, 'euclidean')
        return np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

    def _kmeans_centers(self, X, n_centers, max_iters=100):
        """K-means para encontrar centros"""
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, n_centers, replace=False)
        centers = X[idx].copy()

        for iter_num in range(max_iters):
            distances = cdist(X, centers, 'euclidean')
            labels = np.argmin(distances, axis=1)
            new_centers = np.array([X[labels == i].mean(axis=0)
                                   if np.sum(labels == i) > 0 else centers[i]
                                   for i in range(n_centers)])
            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        return centers

    def _residual_weights_only(self, weights, X, y, centers):
        """Función de residuos optimizando SOLO pesos (centros fijos)"""
        phi = self._gaussian_rbf(X, centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        y_pred = phi @ weights
        return (y_pred - y).flatten()

    def _residual_with_center_refinement(self, params, X, y, centers_init):
        """Función de residuos con refinamiento ligero de centros"""
        n_center_params = self.n_centers * self.n_features

        # Centros = iniciales + pequeño ajuste
        center_deltas = params[:n_center_params].reshape(self.n_centers, self.n_features)
        centers = centers_init + center_deltas

        weights = params[n_center_params:]

        phi = self._gaussian_rbf(X, centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        y_pred = phi @ weights

        return (y_pred - y).flatten()

    def fit(self, X, y, method='lm', verbose=1):
        """
        Entrenar RBF híbrida

        Args:
            X: datos de entrada
            y: valores objetivo
            method: método de optimización ('lm', 'trf', 'dogbox')
            verbose: nivel de verbosidad
        """
        X = np.array(X)
        y = np.array(y)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        self.n_features = X.shape[1]

        print(f"\n{'='*70}")
        if self.refine_centers:
            print("RBF Híbrida: K-means + L-M (con refinamiento de centros)")
        else:
            print("RBF Híbrida: K-means (fijos) + L-M (solo pesos)")
        print(f"{'='*70}")

        # Paso 1: Encontrar centros con K-means
        print("Paso 1: Inicializando centros con K-means...")
        self.centers = self._kmeans_centers(X, self.n_centers)
        print(f"  ✓ {self.n_centers} centros encontrados")

        # Calcular MSE inicial con solución de mínimos cuadrados
        phi_init = self._gaussian_rbf(X, self.centers)
        phi_init = np.hstack([phi_init, np.ones((phi_init.shape[0], 1))])
        weights_init = np.linalg.lstsq(phi_init, y, rcond=None)[0]

        residuals_init = (phi_init @ weights_init - y).flatten()
        mse_init = np.mean(residuals_init ** 2)
        print(f"  MSE inicial (mínimos cuadrados): {mse_init:.6f}")

        # Paso 2: Optimizar pesos (y opcionalmente centros) con L-M
        print(f"\nPaso 2: Optimizando con Levenberg-Marquardt...")
        print(f"  Método: {method}")
        print(f"  Parámetros a optimizar:")

        if self.refine_centers:
            # Optimizar pesos + pequeños ajustes a centros
            print(f"    - Pesos: {self.n_centers + 1}")
            print(f"    - Ajustes de centros: {self.n_centers * self.n_features}")

            # Inicializar: ajustes de centros = 0, pesos = solución lstsq
            params_init = np.concatenate([
                np.zeros(self.n_centers * self.n_features),  # deltas de centros
                weights_init.flatten()
            ])

            result = least_squares(
                self._residual_with_center_refinement,
                params_init,
                args=(X, y, self.centers.copy()),
                method=method,
                verbose=verbose,
                ftol=1e-10,
                xtol=1e-10,
                max_nfev=1000
            )

            # Extraer resultados
            n_center_params = self.n_centers * self.n_features
            center_deltas = result.x[:n_center_params].reshape(self.n_centers, self.n_features)
            self.centers = self.centers + center_deltas
            self.weights = result.x[n_center_params:].reshape(-1, 1)

            print(f"\n  Ajuste máximo en centros: {np.max(np.abs(center_deltas)):.6f}")
            print(f"  Ajuste promedio en centros: {np.mean(np.abs(center_deltas)):.6f}")

        else:
            # Optimizar SOLO pesos (centros fijos)
            print(f"    - Pesos: {self.n_centers + 1}")
            print(f"    - Centros: FIJOS")

            result = least_squares(
                self._residual_weights_only,
                weights_init.flatten(),
                args=(X, y, self.centers),
                method=method,
                verbose=verbose,
                ftol=1e-10,
                xtol=1e-10,
                max_nfev=1000
            )

            self.weights = result.x.reshape(-1, 1)

        # Calcular MSE final
        residuals_final = self._residual_weights_only(
            self.weights.flatten(), X, y, self.centers
        )
        mse_final = np.mean(residuals_final ** 2)

        print(f"\n{'='*70}")
        print("RESULTADOS")
        print(f"{'='*70}")
        print(f"Estado: {result.message}")
        print(f"Éxito: {result.success}")
        print(f"Evaluaciones de función: {result.nfev}")
        print(f"MSE inicial: {mse_init:.10f}")
        print(f"MSE final:   {mse_final:.10f}")
        diferencia = abs(mse_final - mse_init)
        print(f"Diferencia:  {diferencia:.2e}")
        print(f"{'='*70}\n")

        return self

    def predict(self, X):
        """Hacer predicciones"""
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        phi = self._gaussian_rbf(X, self.centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        return (phi @ self.weights).flatten()


def compare_hybrid_methods():
    """Comparar tres enfoques"""

    # Generar datos
    np.random.seed(42)
    X_train = np.linspace(0, 10, 80).reshape(-1, 1)
    y_train = np.sin(X_train).flatten() + 0.15 * np.random.randn(80)

    X_test = np.linspace(0, 10, 300).reshape(-1, 1)
    y_true = np.sin(X_test).flatten()

    print("\n" + "🔵"*35)
    print("COMPARACIÓN DE MÉTODOS HÍBRIDOS")
    print("🔵"*35)

    # Método 1: K-means + Mínimos cuadrados (referencia)
    print("\n" + "="*70)
    print("MÉTODO 1: K-means + Mínimos Cuadrados Lineales (REFERENCIA)")
    print("="*70)

    rbf1 = RBFHybridLM(n_centers=8, sigma=1.0, refine_centers=False)

    # Implementación directa
    X_1d = X_train
    y_1d = y_train.reshape(-1, 1)
    centers_ref = rbf1._kmeans_centers(X_1d, 8)
    phi_ref = rbf1._gaussian_rbf(X_1d, centers_ref)
    phi_ref = np.hstack([phi_ref, np.ones((phi_ref.shape[0], 1))])
    weights_ref = np.linalg.lstsq(phi_ref, y_1d, rcond=None)[0]

    rbf1.centers = centers_ref
    rbf1.weights = weights_ref
    rbf1.n_features = 1

    y_pred1 = rbf1.predict(X_test)
    mse1 = np.mean((rbf1.predict(X_train) - y_train) ** 2)
    print(f"MSE final: {mse1:.10f}")

    # Método 2: K-means (fijos) + L-M para pesos
    print("\n" + "="*70)
    print("MÉTODO 2: K-means (fijos) + Levenberg-Marquardt (solo pesos)")
    print("="*70)

    rbf2 = RBFHybridLM(n_centers=8, sigma=1.0, refine_centers=False)
    rbf2.fit(X_train, y_train, method='lm', verbose=0)
    y_pred2 = rbf2.predict(X_test)
    mse2 = np.mean((rbf2.predict(X_train) - y_train) ** 2)

    # Método 3: K-means + L-M con refinamiento de centros
    print("\n" + "="*70)
    print("MÉTODO 3: K-means + Levenberg-Marquardt (pesos + refinamiento centros)")
    print("="*70)

    rbf3 = RBFHybridLM(n_centers=8, sigma=1.0, refine_centers=True)
    rbf3.fit(X_train, y_train, method='lm', verbose=0)
    y_pred3 = rbf3.predict(X_test)
    mse3 = np.mean((rbf3.predict(X_train) - y_train) ** 2)

    # Visualización
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Fila 1: Predicciones
    methods = [
        (rbf1, y_pred1, mse1, 'K-means + Lstsq\n(Referencia)', 'blue'),
        (rbf2, y_pred2, mse2, 'K-means fijos + L-M\n(solo pesos)', 'purple'),
        (rbf3, y_pred3, mse3, 'K-means + L-M\n(pesos + refinamiento)', 'red')
    ]

    for idx, (rbf, y_pred, mse, title, color) in enumerate(methods):
        ax = axes[0, idx]
        ax.scatter(X_train, y_train, alpha=0.4, s=40, c='gray', label='Datos')
        ax.plot(X_test, y_true, 'g-', label='Verdadera', linewidth=2.5, alpha=0.7)
        ax.plot(X_test, y_pred, color=color, linewidth=2.5, label='Predicción')
        ax.scatter(rbf.centers, np.zeros(rbf.n_centers),
                  c=color, marker='X', s=250, edgecolors='black',
                  linewidths=2, label='Centros', zorder=5)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'{title}\nMSE = {mse:.6f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)

    # Fila 2: Análisis de centros y residuos
    for idx, (rbf, y_pred, mse, title, color) in enumerate(methods):
        ax = axes[1, idx]

        # Graficar residuos
        residuals = rbf.predict(X_train) - y_train
        ax.scatter(X_train, residuals, alpha=0.6, s=50, c=color, edgecolors='black')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        ax.fill_between(X_train.flatten(), -2*np.std(residuals), 2*np.std(residuals),
                        alpha=0.2, color=color)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('Residuo', fontsize=11)
        ax.set_title(f'Residuos\nStd = {np.std(residuals):.4f}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.6, 0.6)

    plt.tight_layout()
    plt.savefig('/home/rodo/1Paper/rbf_hybrid_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Gráfica guardada: rbf_hybrid_comparison.png")

    # Comparación de posiciones de centros
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (rbf, _, _, title, color) in enumerate(methods):
        ax = axes[idx]
        ax.scatter(X_train, y_train, alpha=0.3, s=30, c='gray', label='Datos')
        ax.plot(X_test, y_true, 'g-', linewidth=2, alpha=0.5)

        # Mostrar área de influencia de cada centro
        for i, center in enumerate(rbf.centers):
            x_range = np.linspace(center[0] - 2*rbf.sigma, center[0] + 2*rbf.sigma, 100)
            influence = np.exp(-((x_range - center[0])**2) / (2 * rbf.sigma**2))
            ax.fill_between(x_range, -1.5, 1.5, alpha=0.1, color=color)

        ax.scatter(rbf.centers.flatten(), np.zeros(rbf.n_centers),
                  c=color, marker='X', s=400, edgecolors='black',
                  linewidths=3, label='Centros', zorder=5)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(-0.5, 10.5)

    plt.tight_layout()
    plt.savefig('/home/rodo/1Paper/rbf_centers_influence.png', dpi=150, bbox_inches='tight')
    print("📊 Gráfica guardada: rbf_centers_influence.png")

    # Tabla resumen
    print("\n" + "="*70)
    print("RESUMEN COMPARATIVO")
    print("="*70)
    print(f"{'Método':<45} {'MSE':>15} {'Diferencia':>10}")
    print("-"*70)
    print(f"{'1. K-means + Lstsq (Referencia)':<45} {mse1:>15.10f} {'---':>10}")
    print(f"{'2. K-means fijos + L-M (pesos)':<45} {mse2:>15.10f} {abs(mse2-mse1):>10.2e}")
    print(f"{'3. K-means + L-M (pesos + refinamiento)':<45} {mse3:>15.10f} {abs(mse3-mse1):>10.2e}")
    print("="*70)

    print("\n💡 CONCLUSIONES:")
    print("-"*70)
    if abs(mse2 - mse1) < 1e-6:
        print("✓ Métodos 1 y 2 son equivalentes (como se esperaba)")
        print("  Con centros fijos, L-M y lstsq resuelven el mismo problema lineal")

    if mse3 < mse1:
        mejora = ((mse1 - mse3) / mse1) * 100
        print(f"✓ Método 3 mejora el MSE en {mejora:.3f}% mediante refinamiento de centros")
    else:
        print("✓ El refinamiento de centros no mejora significativamente el resultado")
    print("="*70)

    plt.show()


if __name__ == "__main__":
    compare_hybrid_methods()
