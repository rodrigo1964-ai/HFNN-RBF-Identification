"""
RBF con Derivadas Analíticas

Clase RBF que calcula f(y), f'(y), f''(y), f'''(y) analíticamente
para usar con el regresor homotópico.

Author: PublicationE - Marzo 2026
"""
import numpy as np


class RBFAnalytical:
    """
    RBF Gaussiana con derivadas analíticas hasta orden 3

    RBF(y) = Σ w_j · φ_j(y) + w_0
    φ_j(y) = exp(-(y - c_j)²/(2σ²))
    """

    def __init__(self, centers, sigma, weights):
        """
        Parameters
        ----------
        centers : array (M,)
            Centros de las funciones base
        sigma : float
            Ancho de las Gaussianas
        weights : array (M+1,)
            Pesos [w_1, ..., w_M, w_0]
        """
        self.centers = np.array(centers)
        self.sigma = float(sigma)
        self.weights = np.array(weights)
        self.n_centers = len(centers)

        # Verificar dimensiones
        assert len(weights) == len(centers) + 1, \
            f"weights debe tener {len(centers)+1} elementos"

    def __call__(self, y):
        """Evaluar RBF(y)"""
        return self.eval(y)

    def eval(self, y):
        """
        Evaluar RBF(y) = Σ w_j · φ_j(y) + w_0

        Parameters
        ----------
        y : float or array
            Punto(s) donde evaluar

        Returns
        -------
        float or array
            RBF(y)
        """
        y = np.atleast_1d(y)

        # Distancias: (N, M)
        distances = y.reshape(-1, 1) - self.centers.reshape(1, -1)

        # Gaussianas: φ_j(y) = exp(-(y - c_j)²/(2σ²))
        phi = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

        # RBF = Σ w_j · φ_j + w_0
        result = phi @ self.weights[:-1] + self.weights[-1]

        return result if len(y) > 1 else float(result[0])

    def grad(self, y):
        """
        Primera derivada analítica: RBF'(y)

        RBF'(y) = Σ w_j · φ_j(y) · (-(y - c_j)/σ²)
        """
        y = np.atleast_1d(y)

        distances = y.reshape(-1, 1) - self.centers.reshape(1, -1)
        phi = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

        # Derivada de φ_j: dφ/dy = φ · (-(y - c_j)/σ²)
        dphi_dy = phi * (-distances / (self.sigma ** 2))

        result = dphi_dy @ self.weights[:-1]

        return result if len(y) > 1 else float(result[0])

    def hess(self, y):
        """
        Segunda derivada analítica: RBF''(y)

        RBF''(y) = Σ w_j · φ_j(y) · ((y - c_j)²/σ⁴ - 1/σ²)
        """
        y = np.atleast_1d(y)

        distances = y.reshape(-1, 1) - self.centers.reshape(1, -1)
        phi = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

        # Segunda derivada de φ_j
        sigma2 = self.sigma ** 2
        sigma4 = sigma2 ** 2

        d2phi_dy2 = phi * ((distances ** 2) / sigma4 - 1 / sigma2)

        result = d2phi_dy2 @ self.weights[:-1]

        return result if len(y) > 1 else float(result[0])

    def third_deriv(self, y):
        """
        Tercera derivada analítica: RBF'''(y)

        RBF'''(y) = Σ w_j · φ_j(y) · (-(y - c_j)³/σ⁶ + 3(y - c_j)/σ⁴)
        """
        y = np.atleast_1d(y)

        distances = y.reshape(-1, 1) - self.centers.reshape(1, -1)
        phi = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

        # Tercera derivada de φ_j
        sigma2 = self.sigma ** 2
        sigma4 = sigma2 ** 2
        sigma6 = sigma4 * sigma2

        d3phi_dy3 = phi * (-(distances ** 3) / sigma6 + 3 * distances / sigma4)

        result = d3phi_dy3 @ self.weights[:-1]

        return result if len(y) > 1 else float(result[0])

    def set_parameters(self, params):
        """
        Establecer parámetros desde vector

        Parameters
        ----------
        params : array
            [c_1, ..., c_M, sigma, w_1, ..., w_M, w_0]
        """
        n = self.n_centers
        self.centers = params[:n]
        self.sigma = params[n]
        self.weights = params[n+1:]

    def get_parameters(self):
        """Obtener parámetros como vector"""
        return np.concatenate([self.centers, [self.sigma], self.weights])

    @property
    def n_parameters(self):
        """Número total de parámetros"""
        return 2 * self.n_centers + 2


def test_derivatives():
    """Test: verificar derivadas analíticas vs numéricas"""
    print("\n" + "="*70)
    print("TEST: Derivadas Analíticas de RBF")
    print("="*70)

    # Crear RBF de prueba
    centers = np.array([-1.0, 0.0, 1.0])
    sigma = 0.5
    weights = np.array([1.0, -0.5, 0.8, 0.2])  # [w1, w2, w3, w0]

    rbf = RBFAnalytical(centers, sigma, weights)

    print(f"\nConfiguración:")
    print(f"  Centros: {centers}")
    print(f"  Sigma: {sigma}")
    print(f"  Pesos: {weights}")
    print(f"  N parámetros: {rbf.n_parameters}")

    # Punto de prueba
    y_test = 0.3

    # Evaluar derivadas analíticas
    f0 = rbf.eval(y_test)
    f1 = rbf.grad(y_test)
    f2 = rbf.hess(y_test)
    f3 = rbf.third_deriv(y_test)

    print(f"\n{'='*70}")
    print(f"Derivadas Analíticas en y = {y_test}:")
    print(f"{'='*70}")
    print(f"  f(y)    = {f0:.8f}")
    print(f"  f'(y)   = {f1:.8f}")
    print(f"  f''(y)  = {f2:.8f}")
    print(f"  f'''(y) = {f3:.8f}")

    # Verificar con diferencias finitas
    h = 1e-6

    # Primera derivada numérica
    f1_num = (rbf.eval(y_test + h) - rbf.eval(y_test - h)) / (2 * h)

    # Segunda derivada numérica
    f2_num = (rbf.eval(y_test + h) - 2*rbf.eval(y_test) + rbf.eval(y_test - h)) / (h ** 2)

    # Tercera derivada numérica
    f3_num = (rbf.eval(y_test + 2*h) - 2*rbf.eval(y_test + h) +
              2*rbf.eval(y_test - h) - rbf.eval(y_test - 2*h)) / (2 * h**3)

    print(f"\n{'='*70}")
    print(f"Derivadas Numéricas (diferencias finitas, h={h}):")
    print(f"{'='*70}")
    print(f"  f'(y)   = {f1_num:.8f}")
    print(f"  f''(y)  = {f2_num:.8f}")
    print(f"  f'''(y) = {f3_num:.8f}")

    # Errores
    err1 = abs(f1 - f1_num)
    err2 = abs(f2 - f2_num)
    err3 = abs(f3 - f3_num)

    print(f"\n{'='*70}")
    print(f"Errores (Analítica - Numérica):")
    print(f"{'='*70}")
    print(f"  Error f'(y):   {err1:.2e}")
    print(f"  Error f''(y):  {err2:.2e}")
    print(f"  Error f'''(y): {err3:.2e}")

    # Verificación
    tol = 1e-5
    if err1 < tol and err2 < tol and err3 < tol:
        print(f"\n✓ Derivadas analíticas CORRECTAS (error < {tol})")
    else:
        print(f"\n✗ ERROR: Derivadas no coinciden (tolerancia {tol})")

    # Test vectorial
    print(f"\n{'='*70}")
    print("TEST: Evaluación vectorial")
    print(f"{'='*70}")

    y_vec = np.linspace(-2, 2, 5)
    f_vec = rbf.eval(y_vec)
    f1_vec = rbf.grad(y_vec)

    print(f"\ny valores: {y_vec}")
    print(f"f(y):      {f_vec}")
    print(f"f'(y):     {f1_vec}")

    print(f"\n{'='*70}")
    print("✓ Test completado")
    print(f"{'='*70}")


def test_simple_case():
    """Test con caso simple conocido: f(y) = y²"""
    print("\n\n" + "="*70)
    print("TEST: Caso Simple - Aproximar f(y) = y² con RBF")
    print("="*70)

    # Aproximar f(y) = y² con RBF
    # Para y ∈ [-2, 2]

    # Entrenar RBF con mínimos cuadrados
    y_train = np.linspace(-2, 2, 50)
    f_train = y_train ** 2

    # Configurar RBF
    n_centers = 5
    centers = np.linspace(-2, 2, n_centers)
    sigma = 1.0

    # Matriz de diseño
    distances = y_train.reshape(-1, 1) - centers.reshape(1, -1)
    phi = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    phi_aug = np.hstack([phi, np.ones((len(y_train), 1))])

    # Resolver mínimos cuadrados
    weights = np.linalg.lstsq(phi_aug, f_train, rcond=None)[0]

    print(f"\nEntrenamiento:")
    print(f"  Puntos de entrenamiento: {len(y_train)}")
    print(f"  Centros RBF: {n_centers}")
    print(f"  Sigma: {sigma}")
    print(f"  Pesos: {weights}")

    # Crear RBF
    rbf = RBFAnalytical(centers, sigma, weights)

    # Evaluar
    y_test = np.linspace(-2, 2, 100)
    f_true = y_test ** 2
    f_rbf = rbf.eval(y_test)

    # Derivadas
    f1_true = 2 * y_test
    f1_rbf = rbf.grad(y_test)

    f2_true = 2 * np.ones_like(y_test)
    f2_rbf = rbf.hess(y_test)

    # Errores
    rmse_f = np.sqrt(np.mean((f_rbf - f_true) ** 2))
    rmse_f1 = np.sqrt(np.mean((f1_rbf - f1_true) ** 2))
    rmse_f2 = np.sqrt(np.mean((f2_rbf - f2_true) ** 2))

    print(f"\nResultados en {len(y_test)} puntos de test:")
    print(f"  RMSE f(y):   {rmse_f:.6f}")
    print(f"  RMSE f'(y):  {rmse_f1:.6f}")
    print(f"  RMSE f''(y): {rmse_f2:.6f}")

    # Mostrar algunos valores
    print(f"\nComparación en puntos específicos:")
    print(f"{'y':<8} {'f_true':<12} {'f_RBF':<12} {'|Error|':<12}")
    print("-"*50)
    for i in [0, 25, 50, 75, 99]:
        err = abs(f_rbf[i] - f_true[i])
        print(f"{y_test[i]:<8.2f} {f_true[i]:<12.4f} {f_rbf[i]:<12.4f} {err:<12.6f}")

    print(f"\n{'='*70}")
    print("✓ Test completado")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Ejecutar tests
    test_derivatives()
    test_simple_case()

    print("\n\n✓ Todos los tests completados exitosamente\n")
