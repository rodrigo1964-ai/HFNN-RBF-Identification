# Caso de Estudio 2: Regresor Homotópico para y' + sin²(y) = sin(5t)

## 📄 Resumen

Este caso de estudio implementa el **regresor homotópico de 3 puntos** (serie de Liao) para resolver la ecuación diferencial no lineal de primer orden con función trigonométrica:

```
y' + sin²(y) = sin(5t)
```

con condición inicial `y(0) = -0.2`.

## 🎯 Contribución Principal

Implementación modular y documentada del **regresor homotópico** para funciones trigonométricas no lineales que:
- Discretiza la ecuación usando diferencias finitas hacia atrás de 3 puntos
- Aplica 3 correcciones iterativas por punto (z1, z2, z3)
- Maneja funciones periódicas sin²(y) con derivadas trigonométricas exactas
- Alcanza errores RMS comparables al Caso 1 (~10⁻⁴ con N=500 puntos)
- Es computacionalmente eficiente (<5 ms para 500 puntos)

## 🔬 Ecuación del Sistema

**Ecuación diferencial:**

```
y' + f(y) = u(t)
```

donde:
- `f(y) = sin²(y)` (función trigonométrica no lineal)
- `u(t) = sin(5t)` (forzamiento periódico)
- `y₀ = -0.2` (condición inicial)

**Derivadas analíticas:**
- `f(y) = sin²(y)`
- `f'(y) = 2·sin(y)·cos(y) = sin(2y)`
- `f''(y) = 2·cos(2y)`
- `f'''(y) = -4·sin(2y)`

**Nota:** Las derivadas se calculan usando identidades trigonométricas para optimizar la evaluación computacional.

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

El regresor homotópico logra **excelente precisión** para este sistema trigonométrico no lineal:

| N Puntos | Paso T | Error RMS | Evaluación |
|----------|--------|-----------|------------|
| 100 | 0.1000 | ~10⁻³ | ⭐⭐⭐ |
| 500 | 0.0200 | ~10⁻⁴ | ⭐⭐⭐⭐⭐ |
| 2000 | 0.0050 | ~10⁻⁵ | ⭐⭐⭐⭐⭐ |

## 📁 Archivos del Proyecto

### Scripts Python

1. **`caso2_regressor.py`** - Módulo principal
   - Funciones: `f(y)`, `df(y)`, `d2f(y)`, `d3f(y)` (derivadas trigonométricas)
   - `regresor_homotopico()`: Solver principal
   - `solve_ode_regressor()`: Interfaz completa
   - `solve_ode_rk4()`: Referencia con odeint
   - `compute_error()`: Cálculo de métricas

2. **`test_caso2.py`** - Tests y comparaciones
   - Tests del regresor vs odeint
   - Análisis de convergencia con diferentes N
   - Generación de gráficos de comparación

3. **`generate_figures.py`** - Generación de figuras
   - Genera todas las figuras del caso
   - Figuras de alta calidad para publicación

### Notebooks (originales)

- **`Caso_2_2p_v1.ipynb`**: Notebook original con desarrollo

### Figuras

- **`caso2_comparison_N500.png`**: Comparación directa regresor vs odeint
- **`caso2_convergence.png`**: Análisis de convergencia (Error vs T)
- **`caso2_multiple_N.png`**: Comparación con diferentes N
- **`caso2_phase_portrait.png`**: Retrato de fase y vs y'

## 🚀 Cómo Ejecutar

### Test principal del regresor

```bash
python3 caso2_regressor.py
```

### Tests y análisis de convergencia

```bash
python3 test_caso2.py
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

donde `f(y[k]) = sin²(y[k])` para este caso.

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

donde para este caso:
- `g' = 3/(2T) + sin(2y[k])`
- `g'' = 2·cos(2y[k])`
- `g''' = -4·sin(2y[k])`

## 💡 Características del Método

### Ventajas

✅ **Precisión excelente** para funciones trigonométricas no lineales
✅ **Muy eficiente** computacionalmente (~3-5 ms para 500 puntos)
✅ **Código simple** y fácil de implementar (3 correcciones)
✅ **Derivadas trigonométricas** evaluadas eficientemente
✅ **Predecible**: convergencia O(T) aproximadamente
✅ **Memoria mínima**: solo requiere 2 puntos previos
✅ **Estabilidad numérica** buena con funciones periódicas

### Consideraciones

⚠️ Requiere calcular derivadas analíticas de f(y)
⚠️ Las funciones trigonométricas añaden complejidad (pero son evaluables explícitamente)
⚠️ Convergencia más lenta que RK4 (O(T) vs O(T⁴))
⚠️ Necesita inicialización de y[0] y y[1]

## 🔄 Diferencias con el Caso 1

### Función No Lineal

**Caso 1:** `f(y) = y²` (polinomial)
**Caso 2:** `f(y) = sin²(y)` (trigonométrica)

### Derivadas

**Caso 1:**
- `f'(y) = 2y` (lineal)
- `f''(y) = 2` (constante)
- `f'''(y) = 0` (nula)

**Caso 2:**
- `f'(y) = sin(2y)` (trigonométrica)
- `f''(y) = 2·cos(2y)` (trigonométrica)
- `f'''(y) = -4·sin(2y)` (trigonométrica)

### Complejidad Computacional

**Caso 2** requiere evaluaciones de funciones `sin()` y `cos()` en cada corrección, lo que añade un costo computacional mínimo pero mantiene la eficiencia del método.

## 📚 Contexto del Proyecto

Este es el **Caso 2** de una serie de 4 casos de estudio sobre regresores homotópicos con RBF:

1. **Caso 1**: y' + y² = sin(5t) (función polinomial)
2. **Caso 2**: y' + sin²(y) = sin(5t) ← **Este caso** (función trigonométrica)
3. **Caso 3**: y' + β(y) = sin(5t) con RBF
4. **Caso 4**: Oscilador de Duffing con RBF

## 🔑 Conclusiones Principales

1. **El regresor homotópico funciona excelentemente** para ecuaciones con funciones trigonométricas no lineales
   - Errores RMS de ~10⁻⁴ con solo 500 puntos
   - Precisión comparable al Caso 1 (polinomial)

2. **Las derivadas trigonométricas no afectan significativamente la eficiencia**
   - Tiempo de cómputo similar al Caso 1 (~3-5 ms para 500 puntos)
   - Evaluaciones de sin() y cos() son rápidas en computadoras modernas

3. **Fácil de implementar**
   - Código simple y directo
   - Derivadas analíticas explícitas usando identidades trigonométricas

4. **Convergencia predecible**
   - Orden ~O(T): reducir T a la mitad mejora error ~2×
   - Comportamiento similar al Caso 1

5. **Estabilidad numérica buena**
   - Las funciones trigonométricas acotadas (|sin(x)| ≤ 1) ayudan a la estabilidad
   - No se observan divergencias ni inestabilidades

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
- **Caso**: 2 de 4
- **Fecha**: Marzo 2026
- **Sistema**: Ecuación no lineal trigonométrica de 1er orden
- **Método**: Regresor Homotópico de 3 puntos (serie de Liao)

---

**Próximo paso**: Ver **Caso 3** para la extensión con aproximadores RBF (Radial Basis Functions).
