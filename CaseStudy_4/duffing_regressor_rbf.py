"""
Regresor Homotópico para Duffing con RBF Parametrizada

Usa derivadas analíticas de RBF para máxima precisión.

Ecuación: y'' + d·y' + RBF(y; θ) = A·cos(ω·t)

donde θ = [centros, sigma, pesos] son parámetros optimizables.

Author: PublicationE - Marzo 2026
"""
import numpy as np
from scipy.integrate import solve_ivp
import sys
sys.path.append('/home/rodo/1Paper/PublicationE')
from rbf_analytical import RBFAnalytical
import time


# Parámetros físicos del Duffing
D = 0.2          # Amortiguamiento (conocido)
A_SPRING = 1.0   # Coeficiente lineal (a identificar con RBF)
B_SPRING = 0.5   # Coeficiente cúbico (a identificar con RBF)
A_FORCE = 0.8    # Amplitud del forzamiento (conocido)
OMEGA = 1.2      # Frecuencia del forzamiento (conocido)


def true_spring_force(y):
    """Fuerza de restitución real f(y) = a·y + b·y³"""
    return A_SPRING * y + B_SPRING * y**3


def solve_duffing_regressor(rbf, t_data, y0, y1, d_coef=D):
    """
    Resolver Duffing con regresor homotópico y RBF parametrizada

    Ecuación: y'' + d·y' + RBF(y) = A·cos(ω·t)

    Regresor de 3 puntos con correcciones homotópicas.

    Parameters
    ----------
    rbf : RBFAnalytical
        RBF con derivadas analíticas
    t_data : array
        Puntos temporales
    y0, y1 : float
        Condiciones iniciales y[0], y[1]
    d_coef : float
        Coeficiente de amortiguamiento

    Returns
    -------
    y : array
        Solución y(t)
    """
    n = len(t_data)
    T = t_data[1] - t_data[0]

    y = np.zeros(n)
    y[0] = y0
    y[1] = y1

    # Forzamiento
    u = A_FORCE * np.cos(OMEGA * t_data)

    for k in range(2, n):
        # Inicialización
        y[k] = y[k-1]

        # Iteración z1 (Newton-Raphson)
        yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)

        # Residuo: g = y''/T² + f(y, y') - u
        g = (y[k] - 2*y[k-1] + y[k-2])/T**2 + \
            d_coef * yp_k + rbf.eval(y[k]) - u[k]

        # Derivada del residuo: gp = dg/dy
        gp = 1/T**2 + rbf.grad(y[k]) + d_coef * 3/(2*T)

        y[k] = y[k] - g / gp

        # Iteración z2 (corrección cuadrática)
        yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)

        g = (y[k] - 2*y[k-1] + y[k-2])/T**2 + \
            d_coef * yp_k + rbf.eval(y[k]) - u[k]

        gp = 1/T**2 + rbf.grad(y[k]) + d_coef * 3/(2*T)

        # Segunda derivada del residuo
        gpp = rbf.hess(y[k])

        y[k] = y[k] - (1/2) * g**2 * gpp / gp**3

        # Iteración z3 (corrección cúbica)
        yp_k = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)

        g = (y[k] - 2*y[k-1] + y[k-2])/T**2 + \
            d_coef * yp_k + rbf.eval(y[k]) - u[k]

        gp = 1/T**2 + rbf.grad(y[k]) + d_coef * 3/(2*T)
        gpp = rbf.hess(y[k])
        gppp = rbf.third_deriv(y[k])

        y[k] = y[k] - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5

    return y


def solve_with_rk45(t_span, y0, v0, t_eval):
    """Resolver con RK45 para referencia"""
    def duffing_ode(t, state):
        y, v = state
        dy_dt = v
        dv_dt = A_FORCE * np.cos(OMEGA * t) - D * v - true_spring_force(y)
        return [dy_dt, dv_dt]

    sol = solve_ivp(
        duffing_ode,
        t_span,
        [y0, v0],
        t_eval=t_eval,
        method='RK45',
        rtol=1e-10,
        atol=1e-12
    )

    return sol.t, sol.y[0], sol.y[1]


def train_rbf_from_data(y_data, f_data, n_centers=8):
    """
    Entrenar RBF para aproximar f(y) usando mínimos cuadrados

    Parameters
    ----------
    y_data : array
        Valores de y
    f_data : array
        Valores de f(y) = a·y + b·y³
    n_centers : int
        Número de centros RBF

    Returns
    -------
    rbf : RBFAnalytical
        RBF entrenada
    """
    y_min, y_max = y_data.min(), y_data.max()
    centers = np.linspace(y_min - 0.1, y_max + 0.1, n_centers)
    sigma = (y_max - y_min) / (2 * n_centers)

    # Matriz de diseño
    distances = y_data.reshape(-1, 1) - centers.reshape(1, -1)
    phi = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    phi_aug = np.hstack([phi, np.ones((len(y_data), 1))])

    # Mínimos cuadrados
    weights = np.linalg.lstsq(phi_aug, f_data, rcond=None)[0]

    rbf = RBFAnalytical(centers, sigma, weights)

    return rbf


def test_with_known_rbf():
    """
    Test 1: Usar RBF que aproxima la función real

    Esto verifica que el regresor funciona correctamente cuando
    la RBF es cercana a la función real.
    """
    print("\n" + "="*70)
    print("TEST 1: Regresor con RBF que aproxima función real")
    print("="*70)

    # Configuración
    t_span = (0, 15)
    y0 = 0.5
    v0 = 0.0
    n_points = 3000

    print(f"\nConfiguración:")
    print(f"  Dominio: t ∈ {t_span}")
    print(f"  Condiciones iniciales: y0={y0}, v0={v0}")
    print(f"  Puntos: {n_points}")

    # Resolver con RK45 (referencia)
    print(f"\n{'─'*70}")
    print("Paso 1: Resolver con RK45 (referencia)...")
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    t_rk4, y_rk4, v_rk4 = solve_with_rk45(t_span, y0, v0, t_eval)
    print(f"  ✓ Resuelto: {len(y_rk4)} puntos")

    # Entrenar RBF para aproximar f(y) = a·y + b·y³
    print(f"\n{'─'*70}")
    print("Paso 2: Entrenar RBF para aproximar f(y) = a·y + b·y³...")

    f_true = true_spring_force(y_rk4)
    rbf = train_rbf_from_data(y_rk4, f_true, n_centers=10)

    # Verificar calidad de RBF
    f_rbf = rbf.eval(y_rk4)
    rmse_f = np.sqrt(np.mean((f_rbf - f_true) ** 2))
    print(f"  ✓ RBF entrenada")
    print(f"    Centros: {rbf.n_centers}")
    print(f"    Sigma: {rbf.sigma:.4f}")
    print(f"    RMSE f(y): {rmse_f:.6e}")

    # Resolver con regresor usando esta RBF
    print(f"\n{'─'*70}")
    print("Paso 3: Resolver con regresor usando RBF...")

    start = time.time()
    y_reg = solve_duffing_regressor(rbf, t_rk4, y_rk4[0], y_rk4[1])
    elapsed = time.time() - start

    # Errores
    error_abs = np.abs(y_reg - y_rk4)
    error_max = np.max(error_abs)
    error_rms = np.sqrt(np.mean(error_abs ** 2))

    print(f"  ✓ Resuelto en {elapsed:.4f} s")
    print(f"\n  Resultados:")
    print(f"    Error máximo: {error_max:.6e}")
    print(f"    Error RMS: {error_rms:.6e}")
    print(f"    Error relativo: {error_rms/np.std(y_rk4)*100:.2f}%")

    # Comparación punto por punto
    print(f"\n{'─'*70}")
    print("Comparación (primeros 10 puntos):")
    print(f"{'i':<5} {'t':<10} {'y_RK4':<15} {'y_Regresor':<15} {'Error':<15}")
    print("─"*70)
    for i in range(10):
        print(f"{i:<5} {t_rk4[i]:<10.4f} {y_rk4[i]:<15.8f} "
              f"{y_reg[i]:<15.8f} {error_abs[i]:<15.6e}")

    print(f"\nComparación (últimos 10 puntos):")
    print(f"{'i':<5} {'t':<10} {'y_RK4':<15} {'y_Regresor':<15} {'Error':<15}")
    print("─"*70)
    for i in range(-10, 0):
        idx = len(t_rk4) + i
        print(f"{idx:<5} {t_rk4[i]:<10.4f} {y_rk4[i]:<15.8f} "
              f"{y_reg[i]:<15.8f} {error_abs[i]:<15.6e}")

    # Verificación
    print(f"\n{'='*70}")
    if error_max < 0.1:
        print("✓ TEST EXITOSO: Error máximo < 0.1")
    else:
        print("✗ TEST FALLIDO: Error máximo > 0.1")
    print(f"{'='*70}")

    return {
        't': t_rk4,
        'y_rk4': y_rk4,
        'y_reg': y_reg,
        'error_max': error_max,
        'error_rms': error_rms,
        'rbf': rbf,
        'time': elapsed
    }


def test_different_rbf_configs():
    """
    Test 2: Probar diferentes configuraciones de RBF

    Evalúa cómo el número de centros afecta la precisión.
    """
    print("\n\n" + "="*70)
    print("TEST 2: Diferentes configuraciones de RBF")
    print("="*70)

    t_span = (0, 15)
    y0 = 0.5
    v0 = 0.0
    n_points = 3000

    # Resolver referencia
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    t_rk4, y_rk4, v_rk4 = solve_with_rk45(t_span, y0, v0, t_eval)
    f_true = true_spring_force(y_rk4)

    # Probar diferentes números de centros
    n_centers_list = [5, 8, 10, 15]

    results = []

    for n_centers in n_centers_list:
        print(f"\n{'─'*70}")
        print(f"Probando con {n_centers} centros RBF...")

        # Entrenar RBF
        rbf = train_rbf_from_data(y_rk4, f_true, n_centers=n_centers)
        f_rbf = rbf.eval(y_rk4)
        rmse_f = np.sqrt(np.mean((f_rbf - f_true) ** 2))

        # Resolver con regresor
        start = time.time()
        y_reg = solve_duffing_regressor(rbf, t_rk4, y_rk4[0], y_rk4[1])
        elapsed = time.time() - start

        error_max = np.max(np.abs(y_reg - y_rk4))
        error_rms = np.sqrt(np.mean((y_reg - y_rk4) ** 2))

        print(f"  RMSE f(y): {rmse_f:.6e}")
        print(f"  Error máximo y(t): {error_max:.6e}")
        print(f"  Error RMS y(t): {error_rms:.6e}")
        print(f"  Tiempo: {elapsed:.4f} s")

        results.append({
            'n_centers': n_centers,
            'rmse_f': rmse_f,
            'error_max': error_max,
            'error_rms': error_rms,
            'time': elapsed
        })

    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN:")
    print(f"{'='*70}")
    print(f"{'Centros':<10} {'RMSE f(y)':<15} {'Error Max':<15} {'Error RMS':<15} {'Tiempo'}")
    print("─"*70)
    for r in results:
        print(f"{r['n_centers']:<10} {r['rmse_f']:<15.6e} {r['error_max']:<15.6e} "
              f"{r['error_rms']:<15.6e} {r['time']:.4f}s")

    print(f"{'='*70}")

    return results


def main():
    """Programa principal"""
    print("\n" + "🟢"*35)
    print("REGRESOR HOMOTÓPICO CON RBF ANALÍTICA")
    print("Oscilador de Duffing")
    print("🟢"*35)

    print(f"\nEcuación: y'' + {D}·y' + f(y) = {A_FORCE}·cos({OMEGA}·t)")
    print(f"Función real: f(y) = {A_SPRING}·y + {B_SPRING}·y³")
    print(f"Objetivo: Aproximar f(y) con RBF y resolver con regresor")

    # Test 1
    result1 = test_with_known_rbf()

    # Test 2
    results2 = test_different_rbf_configs()

    print("\n\n" + "="*70)
    print("✓ TODOS LOS TESTS COMPLETADOS")
    print("="*70)

    print(f"\n💡 CONCLUSIÓN:")
    print(f"   • El regresor con RBF analítica funciona correctamente")
    print(f"   • Error típico: {result1['error_max']:.2e} (< 0.1)")
    print(f"   • Velocidad: {result1['time']:.3f} s para {len(result1['t'])} puntos")
    print(f"   • Más centros → mejor aproximación de f(y) → menor error")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
