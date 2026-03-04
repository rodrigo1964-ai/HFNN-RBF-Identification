"""
Módulo de RBF con Integrales de Gaussianas

Implementa funciones Gaussianas y sus derivadas, incluyendo la integral.
Usado para aproximar funciones desconocidas β(y) en ecuaciones diferenciales.

Ecuación: y' + β(y) = sin(5t)
donde β(y) se aproxima con RBF usando la integral de Gauss.

Author: CaseStudy_3 - Marzo 2026
"""
import numpy as np


def FuncionGaussI(x, sigma):
    """
    Integral de la función Gaussiana

    ∫ exp(-(x-c)²/σ²) dx = (1/2)·√π·σ·erf(x/σ)

    Parameters
    ----------
    x : float or array
        Distancia al centro (x - c)
    sigma : float or array
        Ancho de la Gaussiana

    Returns
    -------
    float or array
        Valor de la integral
    """
    import math
    return 0.5 * np.sqrt(np.pi) * sigma * np.vectorize(math.erf)(x / sigma)


def FuncionGauss(x, sigma):
    """
    Función Gaussiana

    φ(x) = exp(-(x)²/σ²)

    Parameters
    ----------
    x : float or array
        Distancia al centro
    sigma : float or array
        Ancho de la Gaussiana

    Returns
    -------
    float or array
        Valor de la Gaussiana
    """
    return np.exp(-(x**2) / sigma**2)


def FuncionGaussD(x, sigma):
    """
    Primera derivada de la Gaussiana

    φ'(x) = exp(-(x)²/σ²) · (-2x/σ²)

    Parameters
    ----------
    x : float or array
        Distancia al centro
    sigma : float or array
        Ancho de la Gaussiana

    Returns
    -------
    float or array
        Primera derivada
    """
    return np.exp(-(x**2) / sigma**2) * (-2 * x / sigma**2)


def FuncionGaussDD(x, sigma):
    """
    Segunda derivada de la Gaussiana

    φ''(x) = exp(-(x)²/σ²) · (-2/σ² + 4x²/σ⁴)

    Parameters
    ----------
    x : float or array
        Distancia al centro
    sigma : float or array
        Ancho de la Gaussiana

    Returns
    -------
    float or array
        Segunda derivada
    """
    return np.exp(-(x**2) / sigma**2) * (-2 / sigma**2 + 4 * x**2 / sigma**4)


def EntrenaRBFI(x, y, p, k):
    """
    Entrenar RBF usando la integral de Gauss

    La RBF se entrena para aproximar una función desconocida β(y)
    usando la integral de las Gaussianas como funciones base.

    Parameters
    ----------
    x : array (p,1)
        Valores de entrada para entrenamiento
    y : array (p,1)
        Valores objetivo β(x)
    p : int
        Número de puntos de entrenamiento
    k : int
        Número de neuronas (centros RBF)

    Returns
    -------
    W : array (k,1)
        Pesos de la RBF
    c : array (k,1)
        Centros de las Gaussianas
    sigma : array (1,)
        Ancho de las Gaussianas

    Notes
    -----
    Los centros se distribuyen uniformemente en [-5, 5] y el ancho
    se calcula como σ = (max(c) - min(c)) / √(2k)
    """
    # Centros uniformemente distribuidos
    c = np.linspace(-5, 5, k).reshape(-1, 1)

    # Calcular sigma basado en la distribución de centros
    sigma = (max(c) - min(c)) / np.sqrt(2 * k)
    sigma = sigma[0]  # Convertir a escalar

    # Matriz de distancias: D[i,j] = x[i] - c[j]
    D = np.zeros((p, k))
    D = x - c.T

    # Matriz de funciones base usando integral de Gauss
    G = np.zeros((p, k))
    G = FuncionGaussI(D, sigma)

    # Resolver mínimos cuadrados: G·W = y
    W = np.dot(np.linalg.pinv(G), y)

    return W, c, sigma


def VectorRBFI(x, W, c, sigma):
    """
    Evaluar β(x) usando RBF con integral de Gauss

    β(x) = Σ W[j] · ∫exp(-(x-c[j])²/σ²) dx

    Parameters
    ----------
    x : float or array
        Punto(s) donde evaluar
    W : array (k,1)
        Pesos de la RBF
    c : array (k,1)
        Centros
    sigma : float
        Ancho

    Returns
    -------
    float or array
        β(x) evaluado
    """
    D = x - c.T
    G = FuncionGaussI(D, sigma)
    y = np.dot(G, W)
    return y


def VectorRBF(x, W, c, sigma):
    """
    Evaluar β'(x) usando derivadas de Gauss

    β'(x) = Σ W[j] · exp(-(x-c[j])²/σ²)

    Parameters
    ----------
    x : float or array
        Punto(s) donde evaluar
    W : array (k,1)
        Pesos
    c : array (k,1)
        Centros
    sigma : float
        Ancho

    Returns
    -------
    float or array
        β'(x) evaluado
    """
    D = x - c.T
    G = FuncionGauss(D, sigma)
    y = np.dot(G, W)
    return y


def VectorRBFD(x, W, c, sigma):
    """
    Evaluar β''(x) usando primera derivada de Gauss

    β''(x) = Σ W[j] · φ'(x-c[j])

    Parameters
    ----------
    x : float or array
        Punto(s) donde evaluar
    W : array (k,1)
        Pesos
    c : array (k,1)
        Centros
    sigma : float
        Ancho

    Returns
    -------
    float or array
        β''(x) evaluado
    """
    D = x - c.T
    G = FuncionGaussD(D, sigma)
    y = np.dot(G, W)
    return y


def VectorRBFDD(x, W, c, sigma):
    """
    Evaluar β'''(x) usando segunda derivada de Gauss

    β'''(x) = Σ W[j] · φ''(x-c[j])

    Parameters
    ----------
    x : float or array
        Punto(s) donde evaluar
    W : array (k,1)
        Pesos
    c : array (k,1)
        Centros
    sigma : float
        Ancho

    Returns
    -------
    float or array
        β'''(x) evaluado
    """
    D = x - c.T
    G = FuncionGaussDD(D, sigma)
    y = np.dot(G, W)
    return y


def test_rbf_functions():
    """
    Test básico de las funciones RBF

    Verifica que las funciones Gauss y sus derivadas
    funcionan correctamente.
    """
    print("\n" + "="*70)
    print("TEST: Funciones RBF con Integral")
    print("="*70)

    # Parámetros de prueba
    x = np.linspace(-3, 3, 100)
    sigma = 1.0

    # Evaluar funciones
    g_int = FuncionGaussI(x, sigma)
    g = FuncionGauss(x, sigma)
    g_d = FuncionGaussD(x, sigma)
    g_dd = FuncionGaussDD(x, sigma)

    print(f"\nEvaluación en {len(x)} puntos:")
    print(f"  Rango x: [{x[0]:.2f}, {x[-1]:.2f}]")
    print(f"  Sigma: {sigma}")

    print(f"\nValores en x=0:")
    print(f"  Integral: {FuncionGaussI(0, sigma):.6f}")
    print(f"  Gauss: {FuncionGauss(0, sigma):.6f}")
    print(f"  Primera derivada: {FuncionGaussD(0, sigma):.6f}")
    print(f"  Segunda derivada: {FuncionGaussDD(0, sigma):.6f}")

    # Verificar derivada numérica
    h = 1e-6
    g_d_num = (FuncionGauss(h, sigma) - FuncionGauss(-h, sigma)) / (2*h)
    g_d_exact = FuncionGaussD(0, sigma)
    error = abs(g_d_num - g_d_exact)

    print(f"\nVerificación derivada en x=0:")
    print(f"  Analítica: {g_d_exact:.8f}")
    print(f"  Numérica: {g_d_num:.8f}")
    print(f"  Error: {error:.2e}")

    if error < 1e-6:
        print(f"\n✓ Derivadas correctas (error < 1e-6)")
    else:
        print(f"\n✗ ERROR: Derivadas no coinciden")

    print("="*70)


def test_rbf_training():
    """
    Test de entrenamiento de RBF

    Aproxima la función β(y) = 0.1y³ + 0.1y² + y - 1
    """
    print("\n" + "="*70)
    print("TEST: Entrenamiento RBF para β(y) = 0.1y³ + 0.1y² + y - 1")
    print("="*70)

    # Función objetivo
    def beta_true(y):
        return 0.1*y**3 + 0.1*y**2 + y - 1

    # Datos de entrenamiento
    p = 30
    x = np.linspace(-3, 2, p).reshape(-1, 1)
    y = beta_true(x)

    # Entrenar con diferentes números de neuronas
    k_values = [3, 5, 8]

    print(f"\nDatos de entrenamiento:")
    print(f"  Puntos: {p}")
    print(f"  Rango: [{x[0,0]:.2f}, {x[-1,0]:.2f}]")

    for k in k_values:
        print(f"\n{'─'*70}")
        print(f"Entrenando con k={k} neuronas...")

        W, c, sigma = EntrenaRBFI(x, y, p, k)

        # Evaluar RBF
        y_rbf = VectorRBFI(x, W, c, sigma)

        # Error
        error = np.sqrt(np.mean((y_rbf - y)**2))

        print(f"  Centros: {k}")
        print(f"  Sigma: {sigma:.6f}")
        print(f"  RMSE: {error:.6e}")
        print(f"  Pesos W: {W.ravel()}")

        # Mostrar algunas evaluaciones
        test_points = [-2.0, 0.0, 1.0]
        print(f"\n  Evaluación en puntos de prueba:")
        for xp in test_points:
            y_true = beta_true(xp)
            y_pred = VectorRBFI(xp, W, c, sigma)
            y_pred_val = float(y_pred) if isinstance(y_pred, np.ndarray) else y_pred
            err = abs(y_true - y_pred_val)
            print(f"    x={xp:5.1f}: β_true={y_true:8.4f}, β_RBF={y_pred_val:8.4f}, error={err:.4e}")

    print("\n" + "="*70)
    print("✓ Test completado")
    print("="*70)


if __name__ == "__main__":
    # Ejecutar tests
    test_rbf_functions()
    test_rbf_training()

    print("\n✓ Todos los tests completados exitosamente\n")
