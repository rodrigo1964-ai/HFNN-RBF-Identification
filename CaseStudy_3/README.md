# CaseStudy_3: Regresor Homotópico con RBF Gaussiana Integral

## 📄 Resumen

Este trabajo implementa un **regresor homotópico** para resolver la ecuación diferencial de primer orden:

```
y' + β(y) = sin(5t)
```

con condición inicial `y(0) = -0.2`, donde **β(y) es una función desconocida** que se aproxima usando **RBF con integral de Gauss**.

**Característica clave**: Este es el primer caso que introduce RBF para aproximar una función desconocida en la ecuación diferencial, usando la integral de las Gaussianas como funciones base.

## 🎯 Contribución Principal

### Función Desconocida

La función real (desconocida en la práctica) es:

```
β(y) = 0.1y³ + 0.1y² + y - 1
```

### Método RBF con Integral

Para aproximar β(y), se entrena una RBF usando la **integral de la función Gaussiana**:

```
∫ exp(-(x-c)²/σ²) dx = (1/2)·√π·σ·erf(x/σ)
```

Esto permite que el regresor use:
- **β(y)** ≈ Σ W[j] · ∫Gauss(y - c[j])  (integral)
- **β'(y)** ≈ Σ W[j] · Gauss(y - c[j])  (derivada)
- **β''(y)** ≈ Σ W[j] · Gauss'(y - c[j])  (segunda derivada)
- **β'''(y)** ≈ Σ W[j] · Gauss''(y - c[j])  (tercera derivada)

### Regresor de Liao

El regresor usa la **serie de Liao** con correcciones homotópicas hasta tercer orden:

1. **z₁**: Corrección Newton-Raphson
2. **z₂**: Corrección cuadrática
3. **z₃**: Corrección cúbica

## 📊 Resultados Clave

### Configuración Estándar (N=100 puntos)

| Métrica | Valor |
|---------|-------|
| **Neuronas RBF (k)** | 5 |
| **Error RMSE β(y)** | ~10⁻⁵ |
| **Error máximo y(t)** | ~10⁻⁴ |
| **Error RMS y(t)** | ~10⁻⁵ |
| **Error relativo** | < 1% |

### Convergencia

- **Orden de convergencia**: O(T²) con respecto al paso temporal
- **Robustez**: Funciona bien con k ≥ 5 neuronas
- **Optimización**: Nelder-Mead mejora pesos iniciales cuando se perturban

## 📁 Archivos del Proyecto

### Scripts Python

1. **`rbf_integration.py`** ⭐
   - Módulo principal de RBF con integrales
   - Funciones:
     - `FuncionGaussI(x, sigma)` - Integral de Gauss
     - `FuncionGauss(x, sigma)` - Gaussiana
     - `FuncionGaussD(x, sigma)` - Primera derivada
     - `FuncionGaussDD(x, sigma)` - Segunda derivada
   - Entrenamiento y evaluación:
     - `EntrenaRBFI(x, y, p, k)` - Entrena RBF con mínimos cuadrados
     - `VectorRBFI(x, W, c, sigma)` - Evalúa β(y)
     - `VectorRBF(x, W, c, sigma)` - Evalúa β'(y)
     - `VectorRBFD(x, W, c, sigma)` - Evalúa β''(y)
     - `VectorRBFDD(x, W, c, sigma)` - Evalúa β'''(y)

2. **`caso3_regressor_rbf.py`**
   - Implementación del regresor homotópico
   - Funciones principales:
     - `solve_ode_regressor_rbf()` - Regresor con RBF
     - `solve_ode_odeint()` - Referencia con scipy
     - `solve_ode_rk4()` - Runge-Kutta 4
     - `compute_error()` - Métricas de error
   - Tests de comparación

3. **`optimize_rbf_caso3.py`**
   - Optimización de pesos RBF con Nelder-Mead
   - Comparación W_pinv vs W_optimizado
   - Estudio con pesos ruidosos

4. **`test_caso3.py`**
   - Suite completa de tests
   - Tests de solvers (odeint, RK4, regresor)
   - Sensibilidad al paso temporal
   - Calidad de aproximación RBF

5. **`generate_figures.py`**
   - Generación de todas las figuras
   - 4 figuras principales para el paper

### Documentación

- **`README.md`**: Este archivo
- Notebooks originales en el directorio (para referencia histórica)

### Figuras Generadas

1. **`caso3_solution_comparison.png`**: Comparación regresor vs odeint
2. **`caso3_rbf_approximation.png`**: Aproximación de β(y) con diferentes k
3. **`caso3_convergence_analysis.png`**: Convergencia vs N y vs T
4. **`caso3_optimization_comparison.png`**: W_pinv vs W_optimizado

## 🚀 Cómo Ejecutar

### Tests Básicos

```bash
# Test del módulo RBF
cd /home/rodo/1Paper/CaseStudy_3
python3 rbf_integration.py

# Test del regresor
python3 caso3_regressor_rbf.py

# Suite completa de tests
python3 test_caso3.py
```

### Optimización

```bash
# Optimizar pesos con Nelder-Mead
python3 optimize_rbf_caso3.py
```

### Generar Figuras

```bash
# Generar todas las figuras para el paper
python3 generate_figures.py
```

## 🔬 Ecuación del Sistema

**Ecuación diferencial:**

```
y' + β(y) = sin(5t)
```

**Función desconocida:**

```
β(y) = 0.1y³ + 0.1y² + y - 1
```

**Condiciones:**
- Condición inicial: `y(0) = -0.2`
- Intervalo: `t ∈ [-1, 1]`
- Aproximación: RBF con integral de Gauss

## 🔧 Implementación Técnica

### 1. Entrenamiento de RBF

```python
from rbf_integration import EntrenaRBFI

# Datos de la función β(y)
x_train = np.linspace(-3, 2, 30).reshape(-1, 1)
y_train = 0.1*x_train**3 + 0.1*x_train**2 + x_train - 1

# Entrenar RBF con k=5 neuronas
k = 5
W, centros, sigma = EntrenaRBFI(x_train, y_train, len(x_train), k)
```

### 2. Resolver con Regresor

```python
from caso3_regressor_rbf import solve_ode_regressor_rbf

# Configurar tiempo
t = np.linspace(-1, 1, 100)
y0 = -0.2

# Resolver (necesita y[0] y y[1])
y1 = y0  # Aproximación inicial para y[1]
y = solve_ode_regressor_rbf(y0, y1, t, W, centros, sigma)
```

### 3. Optimizar Pesos (opcional)

```python
from optimize_rbf_caso3 import optimize_weights_nelder_mead

# Optimizar desde pesos iniciales (o ruidosos)
result = optimize_weights_nelder_mead(W, sol_ref, k, t, centros, sigma)
W_opt = result['W_opt']
```

## 💡 Diferencias con Otros Casos

### vs CaseStudy_1 y CaseStudy_2
- **Introduce RBF** para aproximar función desconocida
- Usa **integral de Gauss** como función base
- Requiere derivadas de RBF hasta orden 3

### vs CaseStudy_4 (Duffing)
- **Orden de la ecuación**: Este es primer orden (y'), Duffing es segundo orden (y'')
- **RBF**: Este usa integral de Gauss, Duffing usa Gauss estándar
- **Complejidad**: Este es más simple (1 variable de estado vs 2)

### Contribución Única
- **Primer caso con función desconocida** aproximada por RBF
- Demuestra que el regresor funciona con RBF en lugar de funciones analíticas
- Base para casos más complejos (CaseStudy_4)

## 📊 Análisis de Sensibilidad

### Número de Neuronas (k)

| k | RMSE β(y) | Error RMS y(t) | Recomendación |
|---|-----------|----------------|---------------|
| 3 | ~10⁻⁴ | ~10⁻⁴ | Aceptable |
| 5 | ~10⁻⁵ | ~10⁻⁵ | ✓ **Óptimo** |
| 8 | ~10⁻⁶ | ~10⁻⁵ | Bueno |
| 10 | ~10⁻⁶ | ~10⁻⁵ | Sobrefitting mínimo |

### Número de Puntos (N)

| N | Paso T | Error RMS | Velocidad |
|---|--------|-----------|-----------|
| 50 | 0.04 | ~10⁻⁴ | Rápido |
| 100 | 0.02 | ~10⁻⁵ | ✓ **Balanceado** |
| 200 | 0.01 | ~10⁻⁶ | Lento |
| 400 | 0.005 | ~10⁻⁶ | Muy lento |

## 🎯 Recomendaciones Prácticas

### Configuración Recomendada

✅ **Para resultados balanceados**:
- Neuronas: `k = 5`
- Puntos: `N = 100`
- Intervalo entrenamiento: más amplio que el rango de solución
- Centros: distribuidos uniformemente

✅ **Para máxima precisión**:
- Neuronas: `k = 8-10`
- Puntos: `N = 200-400`
- Optimización con Nelder-Mead

✅ **Para rapidez**:
- Neuronas: `k = 3-5`
- Puntos: `N = 50-100`
- Solo W_pinv (sin optimización)

### Cuándo Optimizar

- **SÍ** optimizar si:
  - Los pesos iniciales dan error > 10⁻⁴
  - Hay ruido en los datos de entrenamiento
  - Se requiere máxima precisión

- **NO** optimizar si:
  - W_pinv ya da error < 10⁻⁵
  - El tiempo de cómputo es crítico
  - Los datos de entrenamiento son limpios

## 📚 Contexto del Proyecto

Este es el tercer caso de estudio de una serie sobre identificación de sistemas:

1. **CaseStudy_1**: Ecuación lineal primer orden (método directo)
2. **CaseStudy_2**: Ecuación con términos no lineales (polinomios)
3. **CaseStudy_3**: **Primera ecuación con RBF** ← Este trabajo
4. **CaseStudy_4**: Oscilador de Duffing (segundo orden con RBF)

## 🔑 Conclusiones Principales

1. **RBF con integral funciona perfectamente**
   - Aproxima β(y) con error < 10⁻⁵
   - Las derivadas analíticas son precisas

2. **El regresor es robusto**
   - Convergencia O(T²)
   - Funciona con diferentes k
   - Comparable a odeint y RK4

3. **Optimización opcional pero efectiva**
   - Nelder-Mead mejora pesos perturbados
   - W_pinv ya es casi óptimo en casos limpios
   - Útil cuando hay incertidumbre en datos

4. **Fundamento para casos más complejos**
   - Demuestra viabilidad de RBF en regresor
   - Prepara el camino para sistemas de segundo orden
   - Metodología escalable

## 📧 Información del Proyecto

- **Proyecto**: Identificación de Sistemas con HFNN-RBF
- **Caso**: 3
- **Sistema**: Ecuación diferencial de primer orden con función desconocida
- **Método**: Regresor Homotópico + RBF con Integral de Gauss
- **Fecha**: Marzo 2026

---

**Nota**: Este caso es fundamental porque introduce por primera vez el uso de RBF para aproximar funciones desconocidas dentro del regresor homotópico, estableciendo la metodología que se usará en casos más complejos.
