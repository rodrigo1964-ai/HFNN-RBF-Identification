"""
caso2_regressor.py - Regresor Homotópico para y' + sin²(y) = sin(5t)

Implementa el regresor homotópico de 3 puntos (serie de Liao) para resolver
la ecuación diferencial: y' + sin²(y) = sin(5t)

Author: Rodolfo H. Rodrigo - UNSJ
Fecha: Marzo 2026
"""

import numpy as np
from scipy.integrate import odeint


def f(y):
    """
    Función no lineal: f(y) = sin²(y)

    Parameters
    ----------
    y : float or array
        Variable dependiente

    Returns
    -------
    float or array
        f(y) = sin²(y)
    """
    return np.sin(y)**2


def df(y):
    """
    Primera derivada: f'(y) = 2·sin(y)·cos(y) = sin(2y)

    Parameters
    ----------
    y : float or array
        Variable dependiente

    Returns
    -------
    float or array
        f'(y) = sin(2y)
    """
    return np.sin(2*y)


def d2f(y):
    """
    Segunda derivada: f''(y) = 2·cos(2y)

    Parameters
    ----------
    y : float or array
        Variable dependiente

    Returns
    -------
    float or array
        f''(y) = 2·cos(2y)
    """
    return 2*np.cos(2*y)


def d3f(y):
    """
    Tercera derivada: f'''(y) = -4·sin(2y)

    Parameters
    ----------
    y : float or array
        Variable dependiente

    Returns
    -------
    float or array
        f'''(y) = -4·sin(2y)
    """
    return -4*np.sin(2*y)


def ode_modelo(y, t):
    """
    Modelo de la ODE para integradores numéricos estándar (odeint, RK4).

    Ecuación: y' = -sin²(y) + sin(5t)

    Parameters
    ----------
    y : float
        Variable dependiente
    t : float
        Tiempo

    Returns
    -------
    float
        dydt = -sin²(y) + sin(5t)
    """
    return -np.sin(y)**2 + np.sin(5*t)


def regresor_homotopico(T, n, u, y):
    """
    Regresor homotópico de 3 puntos para y' + f(y) = u.

    Aplica 3 correcciones iterativas por punto usando la serie de Liao:
    - z1: Corrección Newton-Raphson
    - z2: Corrección de orden 2 (término cuadrático)
    - z3: Corrección de orden 3 (término cúbico)

    Ecuación discretizada (diferencias finitas hacia atrás, 3 puntos):
        g(y[k]) = (3/2)·y[k]/T - 2·y[k-1]/T + (1/2)·y[k-2]/T + f(y[k]) - u[k]

    Parameters
    ----------
    T : float
        Paso temporal
    n : int
        Número de puntos
    u : array
        Forzamiento u(t) = sin(5t)
    y : array
        Vector solución (y[0] y y[1] deben estar inicializados)

    Returns
    -------
    array
        Solución y[k] para k=0..n-1
    """
    for k in range(2, n):
        # Predicción inicial
        y[k] = y[k-1]

        # Función residual g y su derivada g'
        g = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f(y[k]) - u[k]
        gp = 3/(2*T) + df(y[k])

        # z1: Primera corrección (Newton-Raphson)
        y[k] = y[k] - g / gp

        # Recalcular g, g' para z2
        g = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f(y[k]) - u[k]
        gp = 3/(2*T) + df(y[k])
        gpp = d2f(y[k])

        # z2: Segunda corrección (término de orden 2)
        y[k] = y[k] - (g**2 * gpp) / (2 * gp**3)

        # Recalcular g, g', g'' para z3
        g = (3/2)*y[k]/T - 2*y[k-1]/T + (1/2)*y[k-2]/T + f(y[k]) - u[k]
        gp = 3/(2*T) + df(y[k])
        gpp = d2f(y[k])
        gppp = d3f(y[k])

        # z3: Tercera corrección (término de orden 3)
        y[k] = y[k] - (g**3 * (-gppp*gp + 3*gpp**2)) / (6 * gp**5)

    return y


def solve_ode_regressor(y0, t_span, n, use_rk4_initial=True):
    """
    Resuelve y' + sin²(y) = sin(5t) usando el regresor homotópico.

    Parameters
    ----------
    y0 : float
        Condición inicial y(0)
    t_span : tuple
        (t_inicial, t_final)
    n : int
        Número de puntos
    use_rk4_initial : bool, optional
        Si True, usa odeint para calcular y[1]. Si False, usa extrapolación.
        Por defecto True.

    Returns
    -------
    t : array
        Vector de tiempo
    y : array
        Solución del regresor
    """
    t = np.linspace(t_span[0], t_span[1], n)
    T = t[1] - t[0]
    u = np.sin(5*t)

    # Inicializar solución
    y = np.zeros(n)
    y[0] = y0

    if use_rk4_initial:
        # Usar odeint para calcular y[1] con alta precisión
        sol_initial = odeint(ode_modelo, y0, t[:2])
        y[1] = sol_initial[1, 0]
    else:
        # Aproximación de y[1] usando Euler
        y[1] = y[0] + T * ode_modelo(y[0], t[0])

    # Aplicar regresor
    y = regresor_homotopico(T, n, u, y)

    return t, y


def solve_ode_rk4(y0, t_span, n):
    """
    Resuelve y' + sin²(y) = sin(5t) usando odeint (referencia).

    Parameters
    ----------
    y0 : float
        Condición inicial y(0)
    t_span : tuple
        (t_inicial, t_final)
    n : int
        Número de puntos

    Returns
    -------
    t : array
        Vector de tiempo
    y : array
        Solución de odeint
    """
    t = np.linspace(t_span[0], t_span[1], n)
    sol = odeint(ode_modelo, y0, t)
    return t, sol.ravel()


def compute_error(y_true, y_pred):
    """
    Calcula métricas de error entre dos soluciones.

    Parameters
    ----------
    y_true : array
        Solución de referencia
    y_pred : array
        Solución predicha

    Returns
    -------
    dict
        Diccionario con métricas:
        - max_error: Error máximo absoluto
        - rms_error: Error RMS
        - rel_error: Error relativo (%)
    """
    error_abs = np.abs(y_true - y_pred)
    max_error = np.max(error_abs)
    rms_error = np.sqrt(np.mean(error_abs**2))
    rel_error = (rms_error / np.std(y_true)) * 100

    return {
        'max_error': max_error,
        'rms_error': rms_error,
        'rel_error': rel_error
    }


# ============================================================
# Test principal
# ============================================================
if __name__ == "__main__":
    import time

    print("=" * 70)
    print("Test: Regresor Homotópico para y' + sin²(y) = sin(5t)")
    print("=" * 70)

    # Parámetros
    y0 = -0.2
    t_span = (0, 10)
    n = 500

    # Solución con odeint (referencia)
    print("\n1. Resolviendo con odeint (referencia)...")
    t_rk4, y_rk4 = solve_ode_rk4(y0, t_span, n)

    # Solución con regresor
    print("2. Resolviendo con regresor homotópico...")
    start = time.time()
    t_reg, y_reg = solve_ode_regressor(y0, t_span, n)
    elapsed = time.time() - start

    # Calcular errores
    errors = compute_error(y_rk4, y_reg)

    # Resultados
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"Puntos:           {n}")
    print(f"Paso temporal:    {t_rk4[1] - t_rk4[0]:.6f}")
    print(f"Error máximo:     {errors['max_error']:.4e}")
    print(f"Error RMS:        {errors['rms_error']:.4e}")
    print(f"Error relativo:   {errors['rel_error']:.2f}%")
    print(f"Tiempo cómputo:   {elapsed*1000:.2f} ms")
    print("=" * 70)
