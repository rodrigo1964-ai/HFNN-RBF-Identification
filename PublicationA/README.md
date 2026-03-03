# PublicationA: Identificación de Sistemas con RBF

Este directorio contiene todo el material relacionado con el paper sobre identificación de sistemas no lineales usando Redes de Base Radial (RBF) en ecuaciones diferenciales ordinarias.

## 📄 Documento Principal

- **`rbf_ode_paper.pdf`** - Paper completo (10 páginas, 1.2 MB)
  - Planteamiento del problema
  - Objetivo y metodología
  - Resultados y análisis
  - Conclusiones y referencias

- **`rbf_ode_paper.tex`** - Código fuente LaTeX del paper

## 💻 Código Python

### Scripts Principales

1. **`rbf_example.py`** - Ejemplos básicos de RBF
   - Entrenamiento básico con K-means + mínimos cuadrados
   - Regresión 1D (aproximación de sin(x))
   - Clasificación 2D (problema de espiral)
   - Comparación de diferentes valores de sigma

2. **`ode_rbf_identification.py`** - **Metodología completa de identificación** ⭐
   - PASO 1: Generar/cargar datos experimentales (t_i, y_i)
   - PASO 2: Calcular derivada y' numéricamente
   - PASO 3: Identificar f(y) = sin(t) - y'
   - PASO 4: Entrenar RBF y validar con ODE
   - Visualización completa del proceso

3. **`ode_rbf_data_requirements.py`** - Análisis de sensibilidad
   - Determina cuántos puntos se necesitan
   - Prueba con 10, 15, 20, 30, 40, 50, 75, 100, 150, 200 puntos
   - Análisis de error vs paso temporal
   - Recomendaciones prácticas

4. **`rbf_hybrid_fixed.py`** - Comparación de métodos de entrenamiento
   - K-means + mínimos cuadrados (estándar)
   - K-means + Levenberg-Marquardt (solo pesos)
   - L-M completo (centros + pesos)
   - Análisis de rendimiento

5. **`rbf_levenberg_marquardt.py`** - Implementación con L-M
   - Optimización completa con Levenberg-Marquardt
   - Comparación con método estándar

## 📊 Figuras Incluidas

- **`ode_rbf_identification_complete.png`** - Proceso de identificación completo
  - Datos experimentales y derivada calculada
  - Función f(y) identificada vs real
  - Aproximación con diferentes RBF
  - Validación de soluciones
  - Análisis de errores

- **`ode_rbf_data_requirements.png`** - Análisis de sensibilidad
  - Error vs número de puntos
  - Error vs paso temporal
  - Comparación de soluciones
  - Tabla de recomendaciones

- **`rbf_comprehensive_comparison.png`** - Comparación de métodos
  - Diferentes configuraciones de RBF
  - K-means vs normalizado
  - L-M solo pesos vs completo

## 🎯 Resultados Clave

### Precisión Alcanzada
- Con **50 puntos**: RMSE < 0.001 (R² > 0.9999)
- Con **10 centros RBF**: Aproximación casi perfecta de f(y) = y²

### Recomendaciones
- **Mínimo**: 30 puntos (RMSE < 0.01)
- **Óptimo**: 50-75 puntos (balance precisión/costo)
- **Alta precisión**: 100+ puntos (RMSE < 0.0001)

### Factor Limitante
El error está dominado por la precisión de la derivada numérica y'.
Error ∝ (Δt)²

## 🚀 Cómo Usar

### Ejecutar Identificación Completa
```bash
python3 ode_rbf_identification.py
```

### Análisis de Requisitos de Datos
```bash
python3 ode_rbf_data_requirements.py
```

### Ejemplos Básicos de RBF
```bash
python3 rbf_example.py
```

## 📦 Dependencias

```bash
numpy
scipy
matplotlib
```

Instalar con:
```bash
pip install numpy scipy matplotlib
```

## 🔬 Ecuación Estudiada

```
y' + f(y) = sin(t)
```

donde:
- **f(y) = y²** (función desconocida a identificar)
- **Datos**: {(t_i, y_i)} mediciones experimentales
- **Método**: Identificación mediante RBF

## 📚 Metodología

1. **Datos experimentales**: N mediciones (t_i, y_i)
2. **Derivada numérica**: y'_i ≈ ∇y_i
3. **Identificación**: f(y_i) = sin(t_i) - y'_i
4. **Aproximación RBF**: Entrenar RBF(y) ≈ f(y)
5. **Validación**: Resolver ODE con RBF y comparar

## 📖 Referencias en el Paper

1. Broomhead & Lowe (1988) - RBF fundamentals
2. Poggio & Girosi (1990) - Approximation theory
3. Brunton et al. (2016) - SINDy
4. Raissi et al. (2019) - Physics-Informed Neural Networks

## 📧 Contacto

Este trabajo fue desarrollado como parte del proyecto de identificación de sistemas con RBF.

---

**Fecha de creación**: Marzo 2026
**Última actualización**: Marzo 3, 2026
