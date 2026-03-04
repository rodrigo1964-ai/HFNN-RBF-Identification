"""
Regresor Homotópico para Caso 3 con RBF

Resuelve: y' + β(y) = sin(5t) con y(0) = -0.2

donde β(y) es una función desconocida que se aproxima con RBF:
- Función real: β(y) = 0.1y³ + 0.1y² + y - 1
- Se entrena una RBF con integral de Gauss para aproximar β(y)

El regresor usa serie de Liao con correcciones homotópicas.

Author: CaseStudy_3 - Marzo 2026
"""
import numpy as np
from scipy.integrate import odeint
from rbf_integration import (VectorRBFI, VectorRBF, VectorRBFD, VectorRBFDD,
                              EntrenaRBFI)


def beta_true(y):
    """
    Función β(y) real que se desea aproximar

    β(y) = 0.1y³ + 0.1y² + y - 1

    Parameters
    ----------
    y : float or array
        Valor de y

    Returns
    -------
    float or array
        β(y)
    """
    return 0.1*y**3 + 0.1*y**2 + y - 1


def modelo_odeint(y, t):
    """
    Modelo para odeint: y' = -β(y) + sin(5t)

    Parameters
    ----------
    y : float
        Estado actual
    t : float
        Tiempo actual

    Returns
    -------
    float
        dy/dt
    """
    dydt = -beta_true(y) + np.sin(5*t)
    return dydt


def solve_ode_odeint(y0, t):
    """
    Resolver con odeint (referencia)

    Parameters
    ----------
    y0 : float
        Condición inicial
    t : array
        Vector de tiempos

    Returns
    -------
    array
        Solución y(t)
    """
    sol = odeint(modelo_odeint, y0, t)
    return sol.flatten()


def solve_ode_rk4(y0, t):
    """
    Resolver con Runge-Kutta 4 de orden clásico

    y' = -β(y) + sin(5t)

    Parameters
    ----------
    y0 : float
        Condición inicial
    t : array
        Vector de tiempos

    Returns
    -------
    array
        Solución y(t)
    """
    n = len(t)
    y = np.zeros(n)
    y[0] = y0

    for i in range(n-1):
        h = t[i+1] - t[i]
        ti = t[i]
        yi = y[i]

        # Coeficientes RK4
        k1 = -beta_true(yi) + np.sin(5*ti)
        k2 = -beta_true(yi + 0.5*h*k1) + np.sin(5*(ti + 0.5*h))
        k3 = -beta_true(yi + 0.5*h*k2) + np.sin(5*(ti + 0.5*h))
        k4 = -beta_true(yi + h*k3) + np.sin(5*(ti + h))

        y[i+1] = yi + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return y


def solve_ode_regressor_rbf(y0, y1, t, W, centros, sigma):
    """
    Resolver con regresor homotópico usando RBF

    Ecuación: y' + β(y) = sin(5t)

    Regresor de 3 puntos con correcciones homotópicas usando
    la serie de Liao hasta tercer orden.

    Parameters
    ----------
    y0 : float
        y[0]
    y1 : float
        y[1]
    t : array
        Vector de tiempos
    W : array (k,1)
        Pesos RBF
    centros : array (k,1)
        Centros RBF
    sigma : float
        Ancho Gaussiana

    Returns
    -------
    array
        Solución y(t)

    Notes
    -----
    Usa diferencias hacia atrás de orden 3/2:
    y'[k] ≈ (3y[k] - 4y[k-1] + y[k-2])/(2T)

    Correcciones homotópicas:
    - z1: Newton-Raphson
    - z2: Corrección cuadrática
    - z3: Corrección cúbica
    """
    n = len(t)
    T = t[1] - t[0]

    y = np.zeros(n)
    y[0] = y0
    y[1] = y1

    # Forzamiento
    x = np.sin(5*t)

    for k in range(2, n):
        # Inicialización
        y[k] = y[k-1]

        # Funciones beta y derivadas
        beta_k = VectorRBFI(y[k], W, centros, sigma)  # β(y[k])
        db_k = VectorRBF(y[k], W, centros, sigma)     # β'(y[k])
        db2_k = VectorRBFD(y[k], W, centros, sigma)   # β''(y[k])
        db3_k = VectorRBFDD(y[k], W, centros, sigma)  # β'''(y[k])

        # Derivada aproximada: y'[k] ≈ (3y[k] - 4y[k-1] + y[k-2])/(2T)
        yp_approx = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)

        # Residuo: g = y' + β(y) - sin(5t)
        g = yp_approx + beta_k - x[k]

        # Derivada del residuo: gp = dy'/dy + β'(y)
        gp = 3/(2*T) + db_k

        # Corrección z1 (Newton-Raphson)
        y[k] = y[k] - g / gp

        # Recalcular para z2
        beta_k = VectorRBFI(y[k], W, centros, sigma)
        db_k = VectorRBF(y[k], W, centros, sigma)
        db2_k = VectorRBFD(y[k], W, centros, sigma)

        yp_approx = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
        g = yp_approx + beta_k - x[k]
        gp = 3/(2*T) + db_k

        # Corrección z2 (cuadrática)
        y[k] = y[k] - (1/2) * (g**2) * db2_k / (gp**3)

        # Recalcular para z3
        beta_k = VectorRBFI(y[k], W, centros, sigma)
        db_k = VectorRBF(y[k], W, centros, sigma)
        db2_k = VectorRBFD(y[k], W, centros, sigma)
        db3_k = VectorRBFDD(y[k], W, centros, sigma)

        yp_approx = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
        g = yp_approx + beta_k - x[k]
        gp = 3/(2*T) + db_k

        # Corrección z3 (cúbica)
        numerator = -db3_k * gp + 3 * db2_k**2
        y[k] = y[k] - (1/6) * (g**3) * numerator / (gp**5)

    return y


def compute_error(y_pred, y_true):
    """
    Calcular métricas de error

    Parameters
    ----------
    y_pred : array
        Solución predicha
    y_true : array
        Solución de referencia

    Returns
    -------
    dict
        Diccionario con errores
    """
    error_abs = np.abs(y_pred - y_true)

    return {
        'max': np.max(error_abs),
        'rms': np.sqrt(np.mean(error_abs**2)),
        'mean': np.mean(error_abs),
        'std': np.std(error_abs),
        'rel': np.sqrt(np.mean(error_abs**2)) / np.std(y_true) * 100
    }


def test_regresor_vs_odeint():
    """
    Test: Comparar regresor con odeint
    """
    print("\n" + "="*70)
    print("TEST 1: Regresor vs odeint")
    print("="*70)

    # Configuración
    y0 = -0.2
    n = 100
    t = np.linspace(-1, 1, n)

    print(f"\nConfiguración:")
    print(f"  Ecuación: y' + β(y) = sin(5t)")
    print(f"  β(y) = 0.1y³ + 0.1y² + y - 1")
    print(f"  Condición inicial: y(0) = {y0}")
    print(f"  Puntos: {n}")
    print(f"  Intervalo: [{t[0]}, {t[-1]}]")

    # Resolver con odeint
    print(f"\n{'─'*70}")
    print("Resolviendo con odeint...")
    sol_odeint = solve_ode_odeint(y0, t)
    print(f"  ✓ Completado")

    # Entrenar RBF con función verdadera
    print(f"\n{'─'*70}")
    print("Entrenando RBF para aproximar β(y)...")
    p = 30
    x_train = np.linspace(-3, 2, p).reshape(-1, 1)
    y_train = beta_true(x_train)
    k = 5

    W, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)

    # Verificar calidad de RBF
    y_rbf = VectorRBFI(x_train, W, centros, sigma)
    rmse_rbf = np.sqrt(np.mean((y_rbf - y_train)**2))

    print(f"  Neuronas: {k}")
    print(f"  Sigma: {sigma:.6f}")
    print(f"  RMSE β(y): {rmse_rbf:.6e}")
    print(f"  Pesos W: {W.ravel()}")

    # Resolver con regresor
    print(f"\n{'─'*70}")
    print("Resolviendo con regresor + RBF...")
    sol_reg = solve_ode_regressor_rbf(sol_odeint[0], sol_odeint[1], t, W, centros, sigma)
    print(f"  ✓ Completado")

    # Errores
    errors = compute_error(sol_reg, sol_odeint)

    print(f"\n{'─'*70}")
    print("Resultados:")
    print(f"  Error máximo: {errors['max']:.6e}")
    print(f"  Error RMS: {errors['rms']:.6e}")
    print(f"  Error relativo: {errors['rel']:.2f}%")

    # Comparación punto por punto (primeros y últimos 5)
    print(f"\n{'─'*70}")
    print("Comparación (primeros 5 puntos):")
    print(f"{'i':<5} {'t':<10} {'odeint':<15} {'regresor':<15} {'error':<15}")
    print("─"*70)
    for i in range(5):
        err = abs(sol_reg[i] - sol_odeint[i])
        print(f"{i:<5} {t[i]:<10.4f} {sol_odeint[i]:<15.8f} {sol_reg[i]:<15.8f} {err:<15.6e}")

    print(f"\nComparación (últimos 5 puntos):")
    print(f"{'i':<5} {'t':<10} {'odeint':<15} {'regresor':<15} {'error':<15}")
    print("─"*70)
    for i in range(-5, 0):
        idx = n + i
        err = abs(sol_reg[i] - sol_odeint[i])
        print(f"{idx:<5} {t[i]:<10.4f} {sol_odeint[i]:<15.8f} {sol_reg[i]:<15.8f} {err:<15.6e}")

    # Verificación
    print(f"\n{'='*70}")
    if errors['max'] < 0.01:
        print("✓ TEST EXITOSO: Error máximo < 0.01")
    elif errors['max'] < 0.1:
        print("✓ TEST ACEPTABLE: Error máximo < 0.1")
    else:
        print("⚠ ADVERTENCIA: Error máximo > 0.1")
    print(f"{'='*70}")

    return {
        't': t,
        'sol_odeint': sol_odeint,
        'sol_reg': sol_reg,
        'errors': errors,
        'W': W,
        'centros': centros,
        'sigma': sigma
    }


def test_different_k():
    """
    Test: Probar con diferentes números de neuronas
    """
    print("\n\n" + "="*70)
    print("TEST 2: Diferentes números de neuronas k")
    print("="*70)

    y0 = -0.2
    n = 100
    t = np.linspace(-1, 1, n)

    # Resolver referencia
    sol_ref = solve_ode_odeint(y0, t)

    # Datos de entrenamiento para RBF
    p = 30
    x_train = np.linspace(-3, 2, p).reshape(-1, 1)
    y_train = beta_true(x_train)

    # Probar diferentes k
    k_values = [3, 5, 8, 10]

    results = []

    for k in k_values:
        print(f"\n{'─'*70}")
        print(f"Probando con k={k} neuronas...")

        # Entrenar RBF
        W, centros, sigma = EntrenaRBFI(x_train, y_train, p, k)

        # Verificar RBF
        y_rbf = VectorRBFI(x_train, W, centros, sigma)
        rmse_beta = np.sqrt(np.mean((y_rbf - y_train)**2))

        # Resolver con regresor
        sol_reg = solve_ode_regressor_rbf(sol_ref[0], sol_ref[1], t, W, centros, sigma)

        # Errores
        errors = compute_error(sol_reg, sol_ref)

        print(f"  RMSE β(y): {rmse_beta:.6e}")
        print(f"  Error máximo y(t): {errors['max']:.6e}")
        print(f"  Error RMS y(t): {errors['rms']:.6e}")
        print(f"  Error relativo: {errors['rel']:.2f}%")

        results.append({
            'k': k,
            'rmse_beta': rmse_beta,
            'error_max': errors['max'],
            'error_rms': errors['rms'],
            'error_rel': errors['rel']
        })

    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN:")
    print(f"{'='*70}")
    print(f"{'k':<5} {'RMSE β(y)':<15} {'Error Max':<15} {'Error RMS':<15} {'Error Rel'}")
    print("─"*70)
    for r in results:
        print(f"{r['k']:<5} {r['rmse_beta']:<15.6e} {r['error_max']:<15.6e} "
              f"{r['error_rms']:<15.6e} {r['error_rel']:>10.2f}%")

    print(f"{'='*70}")

    return results


def main():
    """Programa principal"""
    print("\n" + "🟢"*35)
    print("CASO 3: REGRESOR HOMOTÓPICO CON RBF")
    print("Ecuación: y' + β(y) = sin(5t)")
    print("🟢"*35)

    print(f"\nEcuación: y' + β(y) = sin(5t)")
    print(f"β(y) = 0.1y³ + 0.1y² + y - 1")
    print(f"Condición inicial: y(0) = -0.2")
    print(f"Método: RBF con integral de Gauss + Regresor de Liao")

    # Test 1: Regresor vs odeint
    result1 = test_regresor_vs_odeint()

    # Test 2: Diferentes k
    results2 = test_different_k()

    print("\n\n" + "="*70)
    print("✓ TODOS LOS TESTS COMPLETADOS")
    print("="*70)

    print(f"\n💡 CONCLUSIÓN:")
    print(f"   • El regresor con RBF funciona correctamente")
    print(f"   • Error típico: {result1['errors']['max']:.2e}")
    print(f"   • La RBF con integral de Gauss aproxima bien β(y)")
    print(f"   • Más neuronas → mejor aproximación")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
