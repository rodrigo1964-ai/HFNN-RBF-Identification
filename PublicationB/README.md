# PublicationB: Optimización Directa de RBF en ODEs

Este directorio contiene la segunda parte del proyecto sobre identificación de sistemas con RBF, enfocada en el **método de optimización directa** que elimina la necesidad de calcular derivadas numéricas.

## 📄 Documento Principal

- **`rbf_ode_direct_paper.pdf`** - Paper completo (10 páginas, 784 KB) ⭐
  - Motivación y contexto (limitaciones del método tradicional)
  - Reformulación como problema de optimización
  - Metodología de optimización directa
  - Análisis comparativo vs método tradicional
  - Análisis de sensibilidad (10-100 puntos)
  - Guía de selección de método
  - Conclusiones e impacto

- **`rbf_ode_direct_paper.tex`** - Código fuente LaTeX

## 💻 Código Python

### 1. **`ode_rbf_direct_optimization.py`** - Optimización Directa ⭐

Implementación completa del método de optimización directa:

**Características**:
- Parametrización optimizable de RBF (centros, sigma, pesos)
- Dos algoritmos de optimización:
  - **L-BFGS-B**: Optimización local rápida (~ 7-11s)
  - **Evolución Diferencial**: Búsqueda global robusta (~ 40s)
- Comparación automática con método tradicional
- Visualización de resultados

**Cómo usar**:
```bash
python3 ode_rbf_direct_optimization.py
```

**Salida**:
- Comparación de 3 métodos (Despeje, L-BFGS-B, Diff. Evolution)
- Gráficas de soluciones y funciones aprendidas
- Métricas de error y tiempo

### 2. **`ode_rbf_direct_sensitivity.py`** - Análisis de Sensibilidad ⭐

Análisis exhaustivo variando número de puntos:

**Características**:
- Prueba con [10, 15, 20, 30, 40, 50, 75, 100] puntos
- Compara Despeje vs Optimización Directa
- Calcula mejora relativa (%)
- Analiza escalamiento con Δt
- Genera visualizaciones comprehensivas (12 subgráficas)

**Cómo usar**:
```bash
python3 ode_rbf_direct_sensitivity.py
```

**Tiempo de ejecución**: ~2-3 minutos

**Salida**:
- Tabla completa de resultados
- Análisis de mejora vs número de puntos
- Ejemplos visuales con pocos/medios/muchos puntos
- Recomendaciones prácticas

## 📊 Figuras Incluidas

1. **`ode_rbf_direct_optimization.png`** - Comparación de métodos
   - Soluciones y(t) con cada método
   - Funciones f(y) aprendidas
   - Comparación visual de calidad

2. **`ode_rbf_direct_sensitivity.png`** - Análisis completo (12 gráficas)
   - Error vs número de puntos
   - Mejora relativa (%)
   - Tiempo de cómputo
   - Error en f(y)
   - Escalamiento con Δt (log-log)
   - Factor de mejora vs Δt
   - Ejemplos con 10, 30, 100 puntos
   - Funciones f(y) aprendidas en cada caso

## 🎯 Resultados Clave

### Mejora según Número de Puntos

| Puntos | Δt | Mejora Optimización |
|--------|-----|---------------------|
| **10** | 0.556 | **+95.0%** 🥇 |
| **15** | 0.357 | **+97.3%** 🏆 |
| **20** | 0.263 | **+91.3%** 🥈 |
| **30** | 0.172 | **+64.0%** ✅ |
| 50 | 0.102 | +77.1% |
| 100 | 0.051 | +68.3% |

### Comparación con 40 Puntos

| Método | RMSE | Tiempo |
|--------|------|--------|
| Despeje + RBF | 1.646e-03 | 0.007 s |
| **Optimización Directa** | **7.574e-04** | 7.5 s |
| Opt. Global | 3.897e-03 | 43 s |

**Mejora: +54%** en precisión con optimización directa!

## 💡 Guía de Uso

### ¿Cuándo usar Optimización Directa?

✅ **SÍ usar cuando:**
- Tienes **< 30 puntos** de datos
- **Δt > 0.15** (pasos temporales grandes)
- **Datos con ruido** experimental
- **No puedes despejar f(y)** analíticamente
- **Precisión es crítica** (vale la pena el costo computacional)

✅ **NO usar cuando:**
- Tienes **> 50 puntos** bien distribuidos
- **Δt < 0.10** (muestreo denso)
- **Velocidad es crítica** (método 1000x más lento)
- Datos limpios y despeje es posible

## 🔬 Principio del Método

### Método Tradicional (PublicationA)
```
1. Calcular y' numéricamente → Error O(Δt²)
2. Despejar f(y) = sin(t) - y'
3. Entrenar RBF ≈ f(y)

Limitación: Error de derivada DOMINA con Δt grande
```

### Método de Optimización Directa (PublicationB)
```
1. Parametrizar RBF(y; θ) con parámetros desconocidos θ
2. Para cada θ propuesto:
   - Resolver ODE: y' + RBF(y; θ) = sin(t)
   - Evaluar error: ||y_ODE - y_datos||²
3. Optimizar θ para minimizar error

Ventaja: ¡NO calcula derivadas! Evita error O(Δt²)
```

## 📈 Visualización del Insight Principal

**Escalamiento del Error con Δt**:

```
Despeje:        Error ∝ (Δt)²      (cuadrático)
Optimización:   Error ∝ (Δt)^α     (α < 1, sublineal)
```

Cuando Δt es grande → Optimización GANA significativamente

## 🔧 Dependencias

```bash
numpy
scipy
matplotlib
```

Mismo entorno que PublicationA.

## 📚 Relación con PublicationA

**PublicationA** (método base):
- Identificación mediante despeje
- Rápido y eficiente con datos densos
- Limitado por error de derivada

**PublicationB** (método avanzado):
- Identificación mediante optimización
- Robusto con datos escasos
- Elimina limitación de derivada
- Costo computacional mayor pero justificado

## 🎓 Aplicaciones

Este método es especialmente útil en:

1. **Sensores costosos**: Experimentos con pocos puntos de medición
2. **Datos con ruido**: Robustez inherente
3. **Ecuaciones implícitas**: Cuando f no se puede despejar
4. **Physics-Informed ML**: Combinar física con aprendizaje
5. **Identificación de sistemas complejos**: Dinámicas no lineales

## 📊 Comparación de Publicaciones

| Aspecto | PublicationA | PublicationB |
|---------|--------------|--------------|
| **Método** | Despeje + RBF | Optimización Directa |
| **Calcula y'** | ✓ Sí (diferencias finitas) | ✗ No |
| **Velocidad** | 0.007 s | 7-11 s |
| **Precisión (N=15)** | RMSE = 0.0106 | RMSE = 0.0003 |
| **Mejora** | --- | +97% mejor |
| **Mejor con** | N > 50, Δt < 0.10 | N < 30, Δt > 0.15 |
| **Robusto a ruido** | Moderado | Alto |

## 🚀 Ejecución Rápida

```bash
# Comparación básica
python3 ode_rbf_direct_optimization.py

# Análisis completo de sensibilidad (~2-3 min)
python3 ode_rbf_direct_sensitivity.py
```

## 📖 Referencias

- PublicationA: Método tradicional de identificación
- Nocedal & Wright (2006): Algoritmo L-BFGS-B
- Storn & Price (1997): Evolución Diferencial
- Raissi et al. (2019): Physics-Informed Neural Networks
- Chen et al. (2018): Neural ODEs

---

**Fecha de creación**: Marzo 2026
**Parte de**: Proyecto de Identificación de Sistemas con RBF
**Relacionado con**: PublicationA (método base)
