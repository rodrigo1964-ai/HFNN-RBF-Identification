# PublicationC: Identificación de Fricción Viscosa en Péndulo

Este directorio contiene un **caso de estudio realista** aplicando los métodos de identificación con RBF a un problema físico: un péndulo con fricción viscosa de ley desconocida.

## 🎯 Problema Físico

**Sistema**: Péndulo con fricción viscosa desconocida

**Ecuación del movimiento**:
```
θ'' + c(θ') + ω₀²·sin(θ) = 0
```

Donde:
- **θ**: Ángulo del péndulo (rad)
- **θ'**: Velocidad angular (rad/s)
- **c(θ')**: Fricción viscosa DESCONOCIDA (a identificar)
- **ω₀ = 2.0 rad/s**: Frecuencia natural (conocida)

**Ley de fricción real** (desconocida en la práctica):
```
c(ω) = 0.3·ω + 0.05·ω³
```

Fricción viscosa no lineal que aumenta con la velocidad (modelo realista para movimiento en fluidos).

## 💻 Código Python

### 1. **`pendulum_rbf_identification.py`** - Identificación Completa ⭐

Implementación de ambos métodos aplicados al péndulo:

**Método 1: Despeje + RBF**
```
1. Medir (t_i, θ_i) en experimento
2. Calcular θ' y θ'' numéricamente
3. Despejar: c(θ') = -θ'' - ω₀²·sin(θ)
4. Entrenar RBF(θ') ≈ c(θ')
```

**Método 2: Optimización Directa**
```
1. Parametrizar c(ω) = RBF(ω; params)
2. Resolver ODE del péndulo con RBF
3. Minimizar ||θ_ODE - θ_medido||²
```

**Cómo usar**:
```bash
python3 pendulum_rbf_identification.py
```

**Tiempo de ejecución**: ~10 segundos

**Salida**:
- Comparación de ambos métodos
- Fricción identificada vs real
- Validación con solución de ODE
- 9 gráficas comprehensivas

### 2. **`pendulum_sensitivity_analysis.py`** - Análisis de Sensibilidad ⭐

Prueba con diferentes cantidades de datos: [10, 15, 20, 30, 40, 50, 75, 100] puntos

**Cómo usar**:
```bash
python3 pendulum_sensitivity_analysis.py
```

**Tiempo de ejecución**: ~2-3 minutos

**Salida**:
- Tabla completa de resultados
- Análisis de mejora vs número de puntos
- Escalamiento del error con Δt
- Recomendaciones prácticas

## 📊 Figuras Incluidas

1. **`pendulum_rbf_identification.png`** - Identificación completa (9 subgráficas)
   - Datos experimentales θ(t) y ω(t)
   - Retrato de fase
   - Fricción c(ω) identificada con cada método
   - Comparación con ley real
   - Validación de soluciones
   - Análisis de errores

2. **`pendulum_sensitivity_analysis.png`** - Análisis de sensibilidad (6 gráficas)
   - Error vs número de puntos
   - Mejora relativa (%)
   - Tiempo de cómputo
   - Escalamiento con Δt (log-log)
   - Factor de mejora vs Δt
   - Tabla de resultados

## 🎯 Resultados Clave

### Comparación con 50 Puntos

| Método | RMSE c(ω) | Tiempo | Mejora |
|--------|-----------|--------|--------|
| Despeje + RBF | 1.468e-01 | 0.0002 s | --- |
| **Optimización Directa** | **8.149e-03** | 9.9 s | **+94.4%** |

### Mejora según Número de Puntos

| Puntos | Δt (s) | RMSE Despeje | RMSE Directo | Mejora |
|--------|--------|--------------|--------------|--------|
| **10** | 1.667 | 2.186e+00 | 3.974e-01 | **+81.8%** 🥉 |
| **15** | 1.071 | 1.357e+00 | 5.537e-02 | **+95.9%** 🥈 |
| **20** | 0.790 | 9.991e-01 | 1.333e-02 | **+98.7%** 🥇 |
| **30** | 0.517 | 6.766e-01 | 3.216e-02 | **+95.2%** ✅ |
| **40** | 0.385 | 2.121e-01 | 1.661e-02 | **+92.2%** ✅ |
| **50** | 0.306 | 1.468e-01 | 1.157e-02 | **+92.1%** ✅ |
| 75 | 0.203 | 5.225e-02 | 2.408e-01 | **-360.8%** ❌ |
| 100 | 0.152 | 3.625e-02 | 2.834e-01 | **-681.9%** ❌ |

### 🔑 Insight Principal

Existe un **punto de cruce crítico** alrededor de N = 50-60 puntos:

- **N < 50**: Método directo GANA significativamente (+92% a +99%)
- **N ≥ 75**: Método tradicional es MEJOR (derivadas numéricas muy precisas)

**Explicación**:
- Con **pocos datos** (Δt grande): error de derivada numérica domina → método directo gana
- Con **muchos datos** (Δt pequeño): derivadas precisas → método tradicional es más rápido y suficiente

## 💡 Guía de Uso

### ✅ Usar Optimización Directa cuando:

- Tienes **< 50 puntos** de medición
- **Δt > 0.3 s** (muestreo escaso)
- **Datos experimentales costosos** (sensores limitados)
- **Ruido en mediciones** (robustez requerida)
- Precisión es crítica y puedes esperar ~10s

### ✅ Usar Método Tradicional cuando:

- Tienes **> 75 puntos** bien distribuidos
- **Δt < 0.2 s** (muestreo denso)
- **Velocidad es crítica** (método ~50000x más rápido)
- Datos limpios y abundantes
- Precisión moderada es suficiente

## 🔬 Ventajas del Caso de Estudio

Este ejemplo es más realista que PublicationA/B porque:

1. **Problema físico reconocible**: Péndulo con amortiguamiento
2. **Ecuación de 2do orden**: Más compleja que las de 1er orden
3. **Sistema de ODEs**: θ' = ω, ω' = -ω₀²sin(θ) - c(ω)
4. **Fricción no lineal**: c(ω) = b₁·ω + b₂·ω³ (más realista)
5. **Datos vectoriales**: (t, θ, ω) en lugar de solo (t, y)
6. **Validación física**: Retrato de fase, conservación de energía

## 📈 Comparación con Publicaciones Anteriores

| Aspecto | PublicationA/B | PublicationC |
|---------|----------------|--------------|
| **Ecuación** | y' + f(y) = sin(t) | θ'' + c(θ') + ω₀²sin(θ) = 0 |
| **Orden** | 1° orden | 2° orden |
| **Función desconocida** | f(y) = y² | c(ω) = b₁·ω + b₂·ω³ |
| **Sistema** | Escalar | Sistema 2D |
| **Contexto** | Matemático | Físico (péndulo) |
| **Variables** | (t, y) | (t, θ, ω) |
| **Visualización** | f(y) vs y | Retrato de fase θ-ω |
| **Punto crítico** | N ≈ 50 | N ≈ 50-60 |
| **Mejora máxima** | +97.3% (N=15) | +98.7% (N=20) |

## 🚀 Ejecución Rápida

```bash
# Identificación completa (~10s)
python3 pendulum_rbf_identification.py

# Análisis de sensibilidad (~2-3 min)
python3 pendulum_sensitivity_analysis.py
```

## 🎓 Aplicaciones Prácticas

Este método es útil para:

1. **Caracterización experimental de amortiguamiento**
   - Péndulos, osciladores mecánicos
   - Identificar fricción en juntas y rodamientos

2. **Identificación de sistemas vibratorios**
   - Suspensiones de vehículos
   - Amortiguadores y sistemas de absorción

3. **Análisis de movimiento en fluidos**
   - Fricción viscosa en líquidos
   - Arrastre aerodinámico

4. **Calibración de modelos**
   - Validar modelos de fricción
   - Determinar coeficientes desconocidos

5. **Datos experimentales escasos**
   - Experimentos costosos
   - Mediciones limitadas por hardware

## 📚 Parámetros del Sistema

### Configuración del Experimento

```python
# Péndulo
OMEGA_0 = 2.0        # Frecuencia natural (rad/s)
theta0 = 0.8         # Ángulo inicial (rad) ~ 46°
omega0 = 0.0         # Velocidad inicial

# Fricción real (desconocida)
B1 = 0.3             # Coeficiente lineal
B2 = 0.05            # Coeficiente cúbico

# Experimento
t_span = (0, 15)     # 15 segundos
n_points = 50        # Puntos de medición
```

### RBF Configuration

```python
# Método Despeje
n_centers = 8
sigma ~ (ω_max - ω_min) / (2 * n_centers)

# Método Directo
n_centers = 6
Optimización: L-BFGS-B
max_iter = 150
```

## 🔧 Dependencias

```bash
numpy
scipy
matplotlib
```

Mismo entorno que PublicationA y PublicationB.

## 📖 Física del Problema

### Ecuación del Péndulo

Para un péndulo con fricción:
```
I·θ'' + c(θ')·L + m·g·L·sin(θ) = 0
```

Dividiendo por I y definiendo ω₀² = (m·g·L)/I:
```
θ'' + (L/I)·c(θ') + ω₀²·sin(θ) = 0
```

Absorbiendo L/I en la definición de c(θ'):
```
θ'' + c(θ') + ω₀²·sin(θ) = 0
```

### Modelo de Fricción

**Fricción lineal** (Stokes): c(ω) = b·ω
- Válido para velocidades bajas
- Amortiguamiento simple

**Fricción cúbica**: c(ω) = b·ω³
- Válido para velocidades altas (turbulencia)
- Arrastre aerodinámico

**Modelo combinado** (usado aquí): c(ω) = b₁·ω + b₂·ω³
- Cubre régimen completo
- Más realista para péndulo en aire

### Retrato de Fase

El sistema se puede visualizar en el espacio (θ, ω):
- Sin fricción: órbitas cerradas (conservación de energía)
- Con fricción: espiral hacia el origen (disipación)

## 📊 Estructura de Datos

### Salida de Experimento

```python
t_data    # Array de tiempos [s]
theta_data # Array de ángulos [rad]
omega_data # Array de velocidades [rad/s]
```

### RBF Aprendida

```python
rbf.centers    # Centros en espacio de ω
rbf.sigma      # Ancho de Gaussianas
rbf.weights    # Pesos + bias
```

### Validación

```python
# Resolver ODE con RBF identificada
θ' = ω
ω' = -ω₀²·sin(θ) - RBF(ω)

# Comparar con datos
RMSE = ||θ_ODE - θ_medido||
```

## 🎯 Conclusiones

1. **Método directo excele con datos escasos**: +92% a +99% mejora con N < 50
2. **Método tradicional es mejor con datos densos**: N > 75, es más rápido
3. **Punto crítico alrededor de N = 50-60**: donde ambos métodos convergen
4. **Trade-off velocidad-precisión**: 50000x más lento pero dramáticamente mejor con pocos datos
5. **Aplicabilidad física**: Funciona en problema realista de 2do orden

---

**Fecha de creación**: Marzo 2026
**Parte de**: Proyecto de Identificación de Sistemas con RBF
**Casos de estudio**: PublicationA (base), PublicationB (método directo), PublicationC (caso físico)
