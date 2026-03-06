# Caso de Estudio 1: Regresor Homotópico para y' + y² = sin(5t)

## 📄 Resumen

Este caso de estudio implementa el **regresor homotópico de 3 puntos** (serie de Liao) para resolver la ecuación diferencial no lineal de primer orden:

```
y' + y² = sin(5t)
```

con condición inicial `y(0) = -0.2`.

## 🎯 Contribución Principal

Implementación modular y documentada del **regresor homotópico** que:
- Discretiza la ecuación usando diferencias finitas hacia atrás de 3 puntos
- Aplica 3 correcciones iterativas por punto (z1, z2, z3)
- Alcanza errores RMS de ~10⁻⁴ con N=500 puntos
- Es computacionalmente eficiente (<5 ms para 500 puntos)

## 🔬 Ecuación del Sistema

**Ecuación diferencial:**

```
y' + f(y) = u(t)
```

donde:
- `f(y) = y²` (función no lineal)
- `u(t) = sin(5t)` (forzamiento periódico)
- `y₀ = -0.2` (condición inicial)

**Derivadas analíticas:**
- `f(y) = y²`
- `f'(y) = 2y`
- `f''(y) = 2`
- `f'''(y) = 0`

## 📊 Resultados Clave

### Test Principal (N=500 puntos)

| Métrica | Valor |
|---------|-------|
| **Error máximo** | ~10⁻⁴ |
| **Error RMS** | ~10⁻⁴ |
| **Error relativo** | <0.1% |
| **Tiempo de cómputo** | ~3-5 ms |

### Convergencia

El análisis de convergencia muestra un orden aproximado de **p ≈ 0.9-1.0**:

```
Reducir T a la mitad → Error mejora ~2×
```

### Comparación con odeint

El regresor homotópico logra **excelente precisión** para este sistema no lineal de primer orden:

| N Puntos | Paso T | Error RMS | Evaluación |
|----------|--------|-----------|------------|
| 100 | 0.1000 | ~10⁻³ | ⭐⭐⭐ |
| 500 | 0.0200 | ~10⁻⁴ | ⭐⭐⭐⭐⭐ |
| 2000 | 0.0050 | ~10⁻⁵ | ⭐⭐⭐⭐⭐ |

## 📁 Archivos del Proyecto

### Scripts Python

1. **`caso1_regressor.py`**
   - Implementación del regresor homotópico
   - Funciones: `f(y)`, `df(y)`, `d2f(y)`, `d3f(y)` (derivadas analíticas)
   - `regresor_homotopico()`: Solver principal
   - `solve_ode_regressor()`: Interfaz completa
   - `solve_ode_rk4()`: Referencia con odeint
   - `compute_error()`: Cálculo de métricas

2. **`test_caso1.py`**
   - Tests del regresor vs odeint
   - Análisis de convergencia con diferentes N
   - Generación de gráficos de comparación

3. **`generate_figures.py`**
   - Genera todas las figuras del caso
   - Figuras de alta calidad para publicación

### Notebooks (originales)

- **`Caso_1_2p_v1.ipynb`**: Notebook original con desarrollo

### Figuras

- **`caso1_comparison_N500.png`**: Comparación directa regresor vs odeint
- **`caso1_convergence.png`**: Análisis de convergencia (Error vs T)
- **`caso1_multiple_N.png`**: Comparación con diferentes N
- **`caso1_phase_portrait.png`**: Retrato de fase y vs y'

## 🚀 Cómo Ejecutar

### Test principal del regresor

```bash
python3 caso1_regressor.py
```

### Tests y análisis de convergencia

```bash
python3 test_caso1.py
```

### Generar todas las figuras

```bash
python3 generate_figures.py
```

## 🔬 Método del Regresor Homotópico

### Discretización (Diferencias Finitas hacia atrás, 3 puntos)

```
y'[k] ≈ (3y[k] - 4y[k-1] + y[k-2]) / (2T)
```

### Función Residual

```
g(y[k]) = (3/2)·y[k]/T - 2·y[k-1]/T + (1/2)·y[k-2]/T + f(y[k]) - u[k]
```

### Tres Correcciones Iterativas

**z1 - Corrección Newton-Raphson:**
```
y[k] = y[k] - g / g'
```

**z2 - Corrección de orden 2:**
```
y[k] = y[k] - (1/2) · g² · g'' / (g')³
```

**z3 - Corrección de orden 3:**
```
y[k] = y[k] - (1/6) · g³ · (-g'''·g' + 3·g''²) / (g')⁵
```

donde:
- `g' = 3/(2T) + f'(y[k])`
- `g'' = f''(y[k])`
- `g''' = f'''(y[k])`

## 💡 Características del Método

### Ventajas

✅ **Precisión excelente** para sistemas no lineales de 1er orden
✅ **Muy eficiente** computacionalmente (~3-5 ms para 500 puntos)
✅ **Código simple** y fácil de implementar (3 correcciones)
✅ **Derivadas analíticas** calculables explícitamente
✅ **Predecible**: convergencia O(T) aproximadamente
✅ **Memoria mínima**: solo requiere 2 puntos previos

### Consideraciones

⚠️ Requiere calcular derivadas analíticas de f(y)
⚠️ Convergencia más lenta que RK4 (O(T) vs O(T⁴))
⚠️ Necesita inicialización de y[0] y y[1]

## 📚 Contexto del Proyecto

Este es el **Caso 1** de una serie de 4 casos de estudio sobre regresores homotópicos con RBF:

1. **Caso 1**: y' + y² = sin(5t) ← **Este caso**
2. **Caso 2**: y' + sin²(y) = sin(5t)
3. **Caso 3**: y' + β(y) = sin(5t) con RBF
4. **Caso 4**: Oscilador de Duffing con RBF

## 🔑 Conclusiones Principales

1. **El regresor homotópico funciona excelentemente** para ecuaciones no lineales simples de 1er orden
   - Errores RMS de ~10⁻⁴ con solo 500 puntos

2. **Muy eficiente computacionalmente**
   - Solo 3-5 ms para resolver 500 puntos
   - Ideal para aplicaciones en tiempo real

3. **Fácil de implementar**
   - Código simple y directo
   - Derivadas analíticas explícitas

4. **Convergencia predecible**
   - Orden ~O(T): reducir T a la mitad mejora error ~2×

## 💻 Dependencias

```bash
numpy
scipy
matplotlib
```

Instalar con:
```bash
pip install numpy scipy matplotlib
```

## 📧 Información del Proyecto

- **Proyecto**: Identificación de Sistemas con Regresores Homotópicos y RBF
- **Caso**: 1 de 4
- **Fecha**: Marzo 2026
- **Sistema**: Ecuación no lineal de 1er orden
- **Método**: Regresor Homotópico de 3 puntos (serie de Liao)

---

**Próximo paso**: Ver **Caso 2** para la extensión a funciones trigonométricas no lineales.
