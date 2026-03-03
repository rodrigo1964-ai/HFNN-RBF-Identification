"""
RBF Híbrida CORREGIDA: K-means + Levenberg-Marquardt
Con mejor manejo numérico y normalización
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from scipy.spatial.distance import cdist


class RBFHybridCorrected:
    """RBF híbrida con normalización mejorada"""

    def __init__(self, n_centers, sigma=1.0):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None
        self.n_features = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def _gaussian_rbf(self, X, centers):
        """Función RBF gaussiana"""
        distances = cdist(X, centers, 'euclidean')
        return np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

    def _kmeans_centers(self, X, n_centers, max_iters=100):
        """K-means para centros"""
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

    def fit_lstsq(self, X, y, normalize=False):
        """Método estándar: K-means + mínimos cuadrados"""
        X = np.array(X).reshape(-1, 1) if len(np.array(X).shape) == 1 else np.array(X)
        y = np.array(y).reshape(-1, 1) if len(np.array(y).shape) == 1 else np.array(y)

        self.n_features = X.shape[1]

        # Normalizar si se solicita
        if normalize:
            self.X_mean, self.X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
            self.y_mean, self.y_std = y.mean(), y.std() + 1e-8
            X = (X - self.X_mean) / self.X_std
            y = (y - self.y_mean) / self.y_std

        # K-means para centros
        self.centers = self._kmeans_centers(X, self.n_centers)

        # Mínimos cuadrados para pesos
        phi = self._gaussian_rbf(X, self.centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        self.weights = np.linalg.lstsq(phi, y, rcond=None)[0]

        return self

    def fit_lm_weights_only(self, X, y, normalize=False, method='trf'):
        """L-M optimizando SOLO pesos (centros de K-means fijos)"""
        X = np.array(X).reshape(-1, 1) if len(np.array(X).shape) == 1 else np.array(X)
        y = np.array(y).reshape(-1, 1) if len(np.array(y).shape) == 1 else np.array(y)

        self.n_features = X.shape[1]

        # Normalizar si se solicita
        if normalize:
            self.X_mean, self.X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
            self.y_mean, self.y_std = y.mean(), y.std() + 1e-8
            X = (X - self.X_mean) / self.X_std
            y = (y - self.y_mean) / self.y_std

        # K-means para centros (FIJOS)
        self.centers = self._kmeans_centers(X, self.n_centers)

        # Inicialización con lstsq
        phi = self._gaussian_rbf(X, self.centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        weights_init = np.linalg.lstsq(phi, y, rcond=None)[0].flatten()

        print(f"  Inicialización:")
        print(f"    Pesos iniciales (lstsq): min={weights_init.min():.4f}, max={weights_init.max():.4f}")

        # Función de residuos para L-M
        def residuals(w):
            phi_w = self._gaussian_rbf(X, self.centers)
            phi_w = np.hstack([phi_w, np.ones((phi_w.shape[0], 1))])
            return (phi_w @ w.reshape(-1, 1) - y).flatten()

        # Optimizar con L-M
        result = least_squares(
            residuals,
            weights_init,
            method=method,  # 'lm' requiere problema sobredeterminado, 'trf' es más robusto
            verbose=0,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=10000
        )

        self.weights = result.x.reshape(-1, 1)

        print(f"  Resultado L-M:")
        print(f"    Estado: {result.message}")
        print(f"    Evaluaciones: {result.nfev}")
        print(f"    Pesos finales: min={self.weights.min():.4f}, max={self.weights.max():.4f}")
        print(f"    Diferencia con lstsq: {np.linalg.norm(self.weights.flatten() - weights_init):.2e}")

        return self

    def fit_lm_full(self, X, y, normalize=False, method='trf'):
        """L-M optimizando centros Y pesos simultáneamente"""
        X = np.array(X).reshape(-1, 1) if len(np.array(X).shape) == 1 else np.array(X)
        y = np.array(y).reshape(-1, 1) if len(np.array(y).shape) == 1 else np.array(y)

        self.n_features = X.shape[1]

        # Normalizar si se solicita
        if normalize:
            self.X_mean, self.X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
            self.y_mean, self.y_std = y.mean(), y.std() + 1e-8
            X = (X - self.X_mean) / self.X_std
            y = (y - self.y_mean) / self.y_std

        # K-means para inicialización
        centers_init = self._kmeans_centers(X, self.n_centers)

        # Pesos iniciales
        phi = self._gaussian_rbf(X, centers_init)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        weights_init = np.linalg.lstsq(phi, y, rcond=None)[0].flatten()

        # Concatenar parámetros
        params_init = np.concatenate([centers_init.flatten(), weights_init])

        print(f"  Inicialización:")
        print(f"    Parámetros totales: {len(params_init)}")
        print(f"    Centros: {self.n_centers * self.n_features}")
        print(f"    Pesos: {len(weights_init)}")

        # Función de residuos
        def residuals(params):
            n_center_params = self.n_centers * self.n_features
            centers = params[:n_center_params].reshape(self.n_centers, self.n_features)
            weights = params[n_center_params:]

            phi_opt = self._gaussian_rbf(X, centers)
            phi_opt = np.hstack([phi_opt, np.ones((phi_opt.shape[0], 1))])
            return (phi_opt @ weights.reshape(-1, 1) - y).flatten()

        # Optimizar
        result = least_squares(
            residuals,
            params_init,
            method=method,
            verbose=0,
            ftol=1e-10,
            xtol=1e-10,
            max_nfev=10000
        )

        # Extraer resultados
        n_center_params = self.n_centers * self.n_features
        self.centers = result.x[:n_center_params].reshape(self.n_centers, self.n_features)
        self.weights = result.x[n_center_params:].reshape(-1, 1)

        center_movement = np.linalg.norm(self.centers - centers_init)
        print(f"  Resultado L-M:")
        print(f"    Estado: {result.message}")
        print(f"    Evaluaciones: {result.nfev}")
        print(f"    Movimiento de centros: {center_movement:.4f}")

        return self

    def predict(self, X):
        """Predicción"""
        X = np.array(X).reshape(-1, 1) if len(np.array(X).shape) == 1 else np.array(X)

        # Normalizar si fue entrenado con normalización
        if self.X_mean is not None:
            X = (X - self.X_mean) / self.X_std

        phi = self._gaussian_rbf(X, self.centers)
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])
        y_pred = (phi @ self.weights).flatten()

        # Desnormalizar
        if self.y_mean is not None:
            y_pred = y_pred * self.y_std + self.y_mean

        return y_pred


def comprehensive_comparison():
    """Comparación exhaustiva de métodos"""

    # Generar datos
    np.random.seed(42)
    X_train = np.linspace(0, 10, 80)
    y_train = np.sin(X_train) + 0.15 * np.random.randn(80)
    X_test = np.linspace(0, 10, 300)
    y_true = np.sin(X_test)

    print("\n" + "="*80)
    print("COMPARACIÓN EXHAUSTIVA: K-means + Diferentes métodos de optimización")
    print("="*80)

    methods_to_test = [
        ("K-means + Lstsq", 'lstsq', False),
        ("K-means + Lstsq (normalizado)", 'lstsq', True),
        ("K-means fijos + L-M pesos (TRF)", 'lm_weights_trf', False),
        ("K-means fijos + L-M pesos (TRF, norm)", 'lm_weights_trf', True),
        ("L-M completo (TRF)", 'lm_full_trf', False),
        ("L-M completo (TRF, norm)", 'lm_full_trf', True),
    ]

    results = []

    for idx, (name, method_type, normalize) in enumerate(methods_to_test, 1):
        print(f"\n{'─'*80}")
        print(f"MÉTODO {idx}: {name}")
        print(f"{'─'*80}")

        rbf = RBFHybridCorrected(n_centers=8, sigma=1.0)

        if method_type == 'lstsq':
            rbf.fit_lstsq(X_train, y_train, normalize=normalize)
        elif method_type == 'lm_weights_trf':
            rbf.fit_lm_weights_only(X_train, y_train, normalize=normalize, method='trf')
        elif method_type == 'lm_full_trf':
            rbf.fit_lm_full(X_train, y_train, normalize=normalize, method='trf')

        y_pred_train = rbf.predict(X_train)
        y_pred_test = rbf.predict(X_test)

        mse_train = np.mean((y_pred_train - y_train) ** 2)
        mse_test = np.mean((y_pred_test - y_true) ** 2)

        print(f"  MSE entrenamiento: {mse_train:.8f}")
        print(f"  MSE prueba: {mse_test:.8f}")

        results.append({
            'name': name,
            'method': method_type,
            'normalize': normalize,
            'rbf': rbf,
            'y_pred': y_pred_test,
            'mse_train': mse_train,
            'mse_test': mse_test
        })

    # Visualización
    n_methods = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    colors = ['blue', 'cyan', 'purple', 'magenta', 'red', 'orange']

    for idx, res in enumerate(results):
        ax = axes[idx]
        rbf = res['rbf']

        ax.scatter(X_train, y_train, alpha=0.4, s=40, c='gray', label='Datos', zorder=1)
        ax.plot(X_test, y_true, 'g-', label='Verdadera', linewidth=2.5, alpha=0.7, zorder=2)
        ax.plot(X_test, res['y_pred'], color=colors[idx], linewidth=2.5,
                label='Predicción', zorder=3)
        ax.scatter(rbf.centers if rbf.X_mean is None else rbf.centers * rbf.X_std + rbf.X_mean,
                  np.zeros(rbf.n_centers), c=colors[idx], marker='X', s=250,
                  edgecolors='black', linewidths=2, label='Centros', zorder=5)

        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        title = f"{res['name']}\nMSE train={res['mse_train']:.6f}, test={res['mse_test']:.6f}"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.savefig('/home/rodo/1Paper/rbf_comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Gráfica guardada: rbf_comprehensive_comparison.png")

    # Tabla resumen
    print("\n" + "="*80)
    print("TABLA RESUMEN")
    print("="*80)
    print(f"{'Método':<45} {'MSE Train':>15} {'MSE Test':>15}")
    print("-"*80)
    for res in results:
        print(f"{res['name']:<45} {res['mse_train']:>15.8f} {res['mse_test']:>15.8f}")
    print("="*80)

    best_idx = np.argmin([r['mse_test'] for r in results])
    print(f"\n🏆 Mejor método: {results[best_idx]['name']}")
    print(f"   MSE test: {results[best_idx]['mse_test']:.8f}")

    plt.show()


if __name__ == "__main__":
    comprehensive_comparison()
