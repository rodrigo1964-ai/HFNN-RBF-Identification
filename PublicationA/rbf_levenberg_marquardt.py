"""
Red RBF con optimización Levenberg-Marquardt
Optimiza simultáneamente centros y pesos
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


class RBFLevenbergMarquardt:
    """Red RBF optimizada con Levenberg-Marquardt"""

    def __init__(self, n_centers, sigma=1.0):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None
        self.n_features = None

    def _gaussian_rbf(self, X, centers):
        """Función de base radial gaussiana"""
        distances = cdist(X, centers, 'euclidean')
        return np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

    def _forward(self, X, centers, weights):
        """Propagación hacia adelante"""
        phi = self._gaussian_rbf(X, centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        return phi @ weights

    def _residual_function(self, params, X, y):
        """
        Función de residuos para Levenberg-Marquardt

        params: vector concatenado [centers_flat, weights]
        """
        # Extraer centros y pesos del vector de parámetros
        n_center_params = self.n_centers * self.n_features
        centers = params[:n_center_params].reshape(self.n_centers, self.n_features)
        weights = params[n_center_params:]

        # Calcular predicciones
        y_pred = self._forward(X, centers, weights)

        # Retornar residuos
        return (y_pred - y).flatten()

    def fit(self, X, y, method='lm', verbose=2):
        """
        Entrenar usando Levenberg-Marquardt

        Args:
            X: datos de entrada (n_samples, n_features)
            y: valores objetivo (n_samples,)
            method: 'lm' (Levenberg-Marquardt), 'trf' (Trust Region Reflective),
                   'dogbox' (dogleg)
            verbose: nivel de verbosidad (0=silencioso, 1=resumen, 2=por iteración)
        """
        X = np.array(X)
        y = np.array(y)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        self.n_features = X.shape[1]
        n_samples = X.shape[0]

        # Inicialización: usar K-means simple para centros y lstsq para pesos
        idx = np.random.choice(n_samples, self.n_centers, replace=False)
        centers_init = X[idx].copy()

        phi_init = self._gaussian_rbf(X, centers_init)
        phi_init = np.hstack([phi_init, np.ones((phi_init.shape[0], 1))])
        weights_init = np.linalg.lstsq(phi_init, y, rcond=None)[0]

        # Concatenar parámetros iniciales
        params_init = np.concatenate([centers_init.flatten(), weights_init.flatten()])

        print(f"\n{'='*60}")
        print(f"Optimización con Levenberg-Marquardt")
        print(f"{'='*60}")
        print(f"Método: {method}")
        print(f"Número de centros: {self.n_centers}")
        print(f"Sigma: {self.sigma}")
        print(f"Parámetros totales: {len(params_init)}")
        print(f"  - Centros: {self.n_centers * self.n_features}")
        print(f"  - Pesos: {self.n_centers + 1}")

        # Calcular error inicial
        residuals_init = self._residual_function(params_init, X, y)
        mse_init = np.mean(residuals_init ** 2)
        print(f"MSE inicial: {mse_init:.6f}")
        print(f"\nIniciando optimización...\n")

        # Optimizar usando Levenberg-Marquardt
        result = least_squares(
            self._residual_function,
            params_init,
            args=(X, y),
            method=method,
            verbose=verbose,
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=1000
        )

        # Extraer parámetros optimizados
        n_center_params = self.n_centers * self.n_features
        self.centers = result.x[:n_center_params].reshape(self.n_centers, self.n_features)
        self.weights = result.x[n_center_params:].reshape(-1, 1)

        # Calcular error final
        residuals_final = self._residual_function(result.x, X, y)
        mse_final = np.mean(residuals_final ** 2)

        print(f"\n{'='*60}")
        print(f"Resultados de la optimización")
        print(f"{'='*60}")
        print(f"Estado: {result.message}")
        print(f"Éxito: {result.success}")
        print(f"Número de evaluaciones: {result.nfev}")
        print(f"MSE inicial: {mse_init:.6f}")
        print(f"MSE final: {mse_final:.6f}")
        print(f"Mejora: {((mse_init - mse_final) / mse_init * 100):.2f}%")
        print(f"{'='*60}\n")

        return self

    def predict(self, X):
        """Hacer predicciones"""
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        phi = self._gaussian_rbf(X, self.centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        y_pred = phi @ self.weights

        return y_pred.flatten()


class RBFStandard:
    """RBF estándar con mínimos cuadrados lineales (para comparación)"""

    def __init__(self, n_centers, sigma=1.0):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _gaussian_rbf(self, X, centers):
        distances = cdist(X, centers, 'euclidean')
        return np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

    def _kmeans_centers(self, X, n_centers, max_iters=100):
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, n_centers, replace=False)
        centers = X[idx].copy()

        for _ in range(max_iters):
            distances = cdist(X, centers, 'euclidean')
            labels = np.argmin(distances, axis=1)
            new_centers = np.array([X[labels == i].mean(axis=0)
                                   if np.sum(labels == i) > 0 else centers[i]
                                   for i in range(n_centers)])
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        return centers

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        self.centers = self._kmeans_centers(X, self.n_centers)
        phi = self._gaussian_rbf(X, self.centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        self.weights = np.linalg.lstsq(phi, y, rcond=None)[0]
        return self

    def predict(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        phi = self._gaussian_rbf(X, self.centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        return (phi @ self.weights).flatten()


def compare_methods():
    """Comparar RBF estándar vs Levenberg-Marquardt"""

    # Generar datos de prueba
    np.random.seed(42)
    X_train = np.linspace(0, 10, 80).reshape(-1, 1)
    y_train = np.sin(X_train).flatten() + 0.15 * np.random.randn(80)

    X_test = np.linspace(0, 10, 300).reshape(-1, 1)
    y_true = np.sin(X_test).flatten()

    # Método 1: RBF estándar (K-means + mínimos cuadrados)
    print("\n" + "🔵" * 30)
    print("MÉTODO 1: K-means + Mínimos Cuadrados Lineales")
    print("🔵" * 30)

    rbf_standard = RBFStandard(n_centers=8, sigma=1.0)
    rbf_standard.fit(X_train, y_train)
    y_pred_standard = rbf_standard.predict(X_test)
    mse_standard = np.mean((rbf_standard.predict(X_train) - y_train) ** 2)
    print(f"MSE final: {mse_standard:.6f}")

    # Método 2: Levenberg-Marquardt
    print("\n" + "🟢" * 30)
    print("MÉTODO 2: Levenberg-Marquardt (optimización completa)")
    print("🟢" * 30)

    rbf_lm = RBFLevenbergMarquardt(n_centers=8, sigma=1.0)
    rbf_lm.fit(X_train, y_train, method='lm', verbose=1)
    y_pred_lm = rbf_lm.predict(X_test)
    mse_lm = np.mean((rbf_lm.predict(X_train) - y_train) ** 2)

    # Visualización comparativa
    plt.figure(figsize=(16, 5))

    # Gráfica 1: Método estándar
    plt.subplot(1, 3, 1)
    plt.scatter(X_train, y_train, alpha=0.5, label='Datos', s=50)
    plt.plot(X_test, y_true, 'g-', label='Verdadera', linewidth=2.5, alpha=0.7)
    plt.plot(X_test, y_pred_standard, 'b-', label='RBF Estándar', linewidth=2.5)
    plt.scatter(rbf_standard.centers, np.zeros(rbf_standard.n_centers),
               c='blue', marker='X', s=200, edgecolors='black',
               linewidths=2, label='Centros', zorder=5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'Método Estándar\nMSE = {mse_standard:.6f}', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Gráfica 2: Levenberg-Marquardt
    plt.subplot(1, 3, 2)
    plt.scatter(X_train, y_train, alpha=0.5, label='Datos', s=50)
    plt.plot(X_test, y_true, 'g-', label='Verdadera', linewidth=2.5, alpha=0.7)
    plt.plot(X_test, y_pred_lm, 'r-', label='RBF L-M', linewidth=2.5)
    plt.scatter(rbf_lm.centers, np.zeros(rbf_lm.n_centers),
               c='red', marker='X', s=200, edgecolors='black',
               linewidths=2, label='Centros optimizados', zorder=5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'Levenberg-Marquardt\nMSE = {mse_lm:.6f}', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Gráfica 3: Comparación directa
    plt.subplot(1, 3, 3)
    plt.scatter(X_train, y_train, alpha=0.5, label='Datos', s=50, c='gray')
    plt.plot(X_test, y_true, 'g-', label='Verdadera', linewidth=2.5, alpha=0.7)
    plt.plot(X_test, y_pred_standard, 'b--', label=f'Estándar (MSE={mse_standard:.6f})',
             linewidth=2.5, alpha=0.8)
    plt.plot(X_test, y_pred_lm, 'r-', label=f'L-M (MSE={mse_lm:.6f})',
             linewidth=2.5, alpha=0.8)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Comparación de Métodos', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/rodo/1Paper/rbf_comparison_lm.png', dpi=150, bbox_inches='tight')
    print("\n📊 Gráfica guardada en: rbf_comparison_lm.png")

    # Comparación de posiciones de centros
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_train.flatten(), y_train, alpha=0.3, s=30, label='Datos')
    plt.plot(X_test, y_true, 'g-', linewidth=2, alpha=0.5)
    plt.scatter(rbf_standard.centers.flatten(), np.zeros(rbf_standard.n_centers),
               c='blue', marker='X', s=300, edgecolors='black', linewidths=2,
               label='Centros K-means', zorder=5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Posición de Centros: K-means', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(X_train.flatten(), y_train, alpha=0.3, s=30, label='Datos')
    plt.plot(X_test, y_true, 'g-', linewidth=2, alpha=0.5)
    plt.scatter(rbf_lm.centers.flatten(), np.zeros(rbf_lm.n_centers),
               c='red', marker='X', s=300, edgecolors='black', linewidths=2,
               label='Centros optimizados (L-M)', zorder=5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Posición de Centros: Levenberg-Marquardt', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/rodo/1Paper/rbf_centers_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Gráfica guardada en: rbf_centers_comparison.png")

    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN COMPARATIVO")
    print("="*60)
    print(f"{'Método':<30} {'MSE Entrenamiento':>20}")
    print("-"*60)
    print(f"{'K-means + Mínimos Cuadrados':<30} {mse_standard:>20.6f}")
    print(f"{'Levenberg-Marquardt':<30} {mse_lm:>20.6f}")
    print("-"*60)
    mejora = ((mse_standard - mse_lm) / mse_standard * 100)
    print(f"Mejora de L-M sobre estándar: {mejora:+.2f}%")
    print("="*60)

    plt.show()


if __name__ == "__main__":
    compare_methods()
