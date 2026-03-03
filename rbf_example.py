"""
Ejemplo de entrenamiento de Red de Base Radial (RBF)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class RBFNetwork:
    """Red de Base Radial con función gaussiana"""

    def __init__(self, n_centers, sigma=1.0):
        """
        Args:
            n_centers: número de centros RBF
            sigma: parámetro de anchura de las funciones gaussianas
        """
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _gaussian_rbf(self, X, centers):
        """Función de base radial gaussiana"""
        # Calcular distancias euclidianas
        distances = cdist(X, centers, 'euclidean')
        # Aplicar función gaussiana
        return np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

    def _kmeans_centers(self, X, n_centers, max_iters=100):
        """Inicializar centros usando K-means simple"""
        n_samples = X.shape[0]
        # Inicializar centros aleatoriamente
        idx = np.random.choice(n_samples, n_centers, replace=False)
        centers = X[idx].copy()

        for _ in range(max_iters):
            # Asignar puntos al centro más cercano
            distances = cdist(X, centers, 'euclidean')
            labels = np.argmin(distances, axis=1)

            # Actualizar centros
            new_centers = np.array([X[labels == i].mean(axis=0)
                                   if np.sum(labels == i) > 0 else centers[i]
                                   for i in range(n_centers)])

            # Verificar convergencia
            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        return centers

    def fit(self, X, y):
        """
        Entrenar la red RBF

        Args:
            X: datos de entrada (n_samples, n_features)
            y: valores objetivo (n_samples,) o (n_samples, n_outputs)
        """
        # Asegurar que X y y son arrays numpy
        X = np.array(X)
        y = np.array(y)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Encontrar centros usando K-means
        self.centers = self._kmeans_centers(X, self.n_centers)

        # Calcular activaciones RBF
        phi = self._gaussian_rbf(X, self.centers)

        # Añadir término de bias
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])

        # Resolver mínimos cuadrados para encontrar pesos
        # phi @ weights = y
        self.weights = np.linalg.lstsq(phi, y, rcond=None)[0]

        return self

    def predict(self, X):
        """Hacer predicciones"""
        X = np.array(X)

        # Calcular activaciones RBF
        phi = self._gaussian_rbf(X, self.centers)

        # Añadir término de bias
        phi = np.hstack([phi, np.ones((phi.shape[0], 1))])

        # Predicción
        y_pred = phi @ self.weights

        return y_pred.flatten() if y_pred.shape[1] == 1 else y_pred


def example_1d_regression():
    """Ejemplo 1: Regresión 1D con función no lineal"""
    print("=" * 60)
    print("Ejemplo 1: Regresión 1D")
    print("=" * 60)

    # Generar datos de entrenamiento
    np.random.seed(42)
    X_train = np.linspace(0, 10, 100).reshape(-1, 1)
    y_train = np.sin(X_train).flatten() + 0.1 * np.random.randn(100)

    # Datos de prueba (más densos para visualización suave)
    X_test = np.linspace(0, 10, 300).reshape(-1, 1)
    y_true = np.sin(X_test).flatten()

    # Entrenar red RBF
    rbf = RBFNetwork(n_centers=10, sigma=1.0)
    rbf.fit(X_train, y_train)

    # Predicciones
    y_pred = rbf.predict(X_test)

    # Calcular error
    y_pred_train = rbf.predict(X_train)
    mse_train = np.mean((y_train - y_pred_train) ** 2)
    print(f"MSE en entrenamiento: {mse_train:.6f}")

    # Visualizar resultados
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, alpha=0.5, label='Datos de entrenamiento')
    plt.plot(X_test, y_true, 'g-', label='Función verdadera', linewidth=2)
    plt.plot(X_test, y_pred, 'r-', label='Predicción RBF', linewidth=2)
    plt.scatter(rbf.centers, np.zeros(rbf.n_centers),
                c='red', marker='x', s=200, linewidths=3,
                label='Centros RBF', zorder=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Red RBF: Aproximación de sin(x)')
    plt.grid(True, alpha=0.3)

    # Visualizar funciones de base radial
    plt.subplot(1, 2, 2)
    X_viz = np.linspace(0, 10, 300).reshape(-1, 1)
    phi = rbf._gaussian_rbf(X_viz, rbf.centers)
    for i in range(rbf.n_centers):
        plt.plot(X_viz, phi[:, i], alpha=0.6, label=f'RBF {i+1}' if i < 3 else None)
    plt.xlabel('x')
    plt.ylabel('Activación')
    plt.title('Funciones de Base Radial (Gaussianas)')
    plt.grid(True, alpha=0.3)
    if rbf.n_centers <= 3:
        plt.legend()

    plt.tight_layout()
    plt.savefig('/home/rodo/1Paper/rbf_1d_example.png', dpi=150, bbox_inches='tight')
    print("Gráfica guardada en: rbf_1d_example.png")
    plt.show()


def example_2d_classification():
    """Ejemplo 2: Clasificación 2D"""
    print("\n" + "=" * 60)
    print("Ejemplo 2: Clasificación 2D")
    print("=" * 60)

    # Generar datos en espiral
    np.random.seed(42)
    n_samples = 200
    theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi

    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + 0.5 * np.random.randn(n_samples, 2)

    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + 0.5 * np.random.randn(n_samples, 2)

    X_train = np.vstack([x_a, x_b])
    y_train = np.hstack([np.ones(n_samples), np.zeros(n_samples)])

    # Entrenar red RBF
    rbf = RBFNetwork(n_centers=20, sigma=2.0)
    rbf.fit(X_train, y_train)

    # Predicciones
    y_pred_train = rbf.predict(X_train)
    y_pred_binary = (y_pred_train > 0.5).astype(int)
    accuracy = np.mean(y_pred_binary == y_train)
    print(f"Precisión en entrenamiento: {accuracy * 100:.2f}%")

    # Crear malla para visualización de frontera de decisión
    h = 0.5
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = rbf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Visualizar
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    plt.colorbar(label='Predicción')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                c='red', edgecolors='k', label='Clase 1', alpha=0.7)
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                c='blue', edgecolors='k', label='Clase 0', alpha=0.7)
    plt.scatter(rbf.centers[:, 0], rbf.centers[:, 1],
                c='green', marker='X', s=200, edgecolors='black',
                linewidths=2, label='Centros RBF', zorder=5)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Clasificación con Red RBF (Espiral)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                c='red', edgecolors='k', label='Clase 1', alpha=0.7)
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                c='blue', edgecolors='k', label='Clase 0', alpha=0.7)
    plt.scatter(rbf.centers[:, 0], rbf.centers[:, 1],
                c='green', marker='X', s=200, edgecolors='black',
                linewidths=2, label='Centros RBF', zorder=5)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Frontera de Decisión')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/rodo/1Paper/rbf_2d_classification.png', dpi=150, bbox_inches='tight')
    print("Gráfica guardada en: rbf_2d_classification.png")
    plt.show()


def example_comparison_sigma():
    """Ejemplo 3: Comparación de diferentes valores de sigma"""
    print("\n" + "=" * 60)
    print("Ejemplo 3: Efecto del parámetro sigma")
    print("=" * 60)

    # Generar datos
    np.random.seed(42)
    X_train = np.linspace(0, 10, 50).reshape(-1, 1)
    y_train = np.sin(X_train).flatten() + 0.2 * np.random.randn(50)
    X_test = np.linspace(0, 10, 300).reshape(-1, 1)
    y_true = np.sin(X_test).flatten()

    sigmas = [0.5, 1.0, 2.0, 5.0]

    plt.figure(figsize=(15, 10))

    for idx, sigma in enumerate(sigmas, 1):
        rbf = RBFNetwork(n_centers=8, sigma=sigma)
        rbf.fit(X_train, y_train)
        y_pred = rbf.predict(X_test)

        mse = np.mean((rbf.predict(X_train) - y_train) ** 2)

        plt.subplot(2, 2, idx)
        plt.scatter(X_train, y_train, alpha=0.5, label='Datos')
        plt.plot(X_test, y_true, 'g-', label='Verdadera', linewidth=2, alpha=0.7)
        plt.plot(X_test, y_pred, 'r-', label='RBF', linewidth=2)
        plt.scatter(rbf.centers, np.zeros(rbf.n_centers),
                   c='red', marker='x', s=150, linewidths=3,
                   label='Centros', zorder=5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'σ = {sigma} (MSE = {mse:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-2, 2)

    plt.tight_layout()
    plt.savefig('/home/rodo/1Paper/rbf_sigma_comparison.png', dpi=150, bbox_inches='tight')
    print("Gráfica guardada en: rbf_sigma_comparison.png")
    plt.show()


if __name__ == "__main__":
    print("\n🔵 Ejemplos de Entrenamiento de Red de Base Radial (RBF)\n")

    # Ejecutar ejemplos
    example_1d_regression()
    example_2d_classification()
    example_comparison_sigma()

    print("\n" + "=" * 60)
    print("✓ Todos los ejemplos completados exitosamente")
    print("=" * 60)
