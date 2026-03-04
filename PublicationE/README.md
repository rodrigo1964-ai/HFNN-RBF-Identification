# PublicationE: Regresor Homotópico con RBF para el Oscilador de Duffing

## 📄 Resumen

Este trabajo presenta un **regresor homotópico de tercer orden** para resolver numéricamente el oscilador de Duffing forzado usando Redes de Base Radial (RBF) con derivadas analíticas. El método permite obtener soluciones precisas sin recurrir a integradores numéricos tradicionales (RK45, etc.).

## 🎯 Contribución Principal

**Regresor homotópico** que:
- Discretiza la ecuación diferencial usando diferencias finitas de orden superior
- Utiliza RBF con derivadas analíticas ($f$, $f'$, $f''$, $f'''$)
- Aplica correcciones homotópicas iterativas para mejorar precisión
- Logra errores RMS de $\sim 4.09 \times 10^{-2}$ en 0.31 s (N=3000)

## 📊 Resultados Clave

### Test Principal (N=3000 puntos)

| Métrica | Valor |
|---------|-------|
| **Error máximo** | $9.05 \times 10^{-2}$ |
| **Error RMS** | $4.09 \times 10^{-2}$ |
| **Error relativo** | 4.00% |
| **Tiempo de cómputo** | 0.309 s |

### Comparación: Regresor vs Tradicional

El análisis muestra **tres regímenes**:

#### 🟢 **Datos Escasos (N ≤ 20)**
- **Mejora promedio**: +98.8%
- **Ganador**: 🏆 **Regresor (claramente superior)**
- El método tradicional falla catastróficamente (errores > 10¹)

#### 🟡 **Datos Moderados (20 < N ≤ 30)**
- **Mejora promedio**: +24.5%
- **Ganador**: ⚖️ Zona de transición

#### 🔴 **Datos Abundantes (N > 30)**
- **Mejora promedio**: -22.4%
- **Ganador**: 🏆 **Método Tradicional**

### Punto de Transición

```
N ≈ 20-25 puntos
T ≈ 0.5-0.6 s
```

## 📁 Archivos del Proyecto

### Scripts Python

1. **`rbf_analytical.py`**
   - Implementación de RBF con derivadas analíticas hasta orden 3
   - Funciones Gaussianas: $\phi_j(y) = \exp(-(y - c_j)^2/(2\sigma^2))$

2. **`duffing_regressor_rbf.py`**
   - Regresor homotópico principal
   - Solución numérica de la ecuación de Duffing
   - Test con RBF que aproxima la función real

3. **`optimize_rbf_regressor.py`**
   - Comparación entre método tradicional (despeje + RBF) y optimización directa
   - Análisis de sensibilidad con diferentes N

4. **`sensitivity_analysis_regressor.py`**
   - Análisis completo de sensibilidad (N = 10 a 50)
   - Identificación de regímenes óptimos

5. **`test_duffing_regressor.py`**
   - Tests del regresor vs RK4
   - Diferentes pasos temporales

6. **`generate_figures.py`**
   - Generación de todas las figuras del paper

### Documentos

- **`duffing_regressor_paper.tex`**: Código fuente LaTeX del paper
- **`duffing_regressor_paper.pdf`**: Paper compilado (8 páginas)
- **`README.md`**: Este archivo

### Figuras

- **`duffing_regressor_comparison_N10.png`**: Comparación con datos escasos
- **`duffing_sensitivity_analysis.png`**: Análisis de sensibilidad (Error vs N)
- **`duffing_rbf_comparison.png`**: Aproximación de f(y) con diferentes N

## 🚀 Cómo Ejecutar

### Generar resultados completos

```bash
# Test principal del regresor
python3 duffing_regressor_rbf.py

# Optimización y comparación
python3 optimize_rbf_regressor.py

# Análisis de sensibilidad completo
python3 sensitivity_analysis_regressor.py

# Generar figuras para el paper
python3 generate_figures.py
```

### Compilar el paper

```bash
pdflatex duffing_regressor_paper.tex
pdflatex duffing_regressor_paper.tex  # Segunda pasada para referencias
```

## 🔬 Ecuación del Sistema

**Oscilador de Duffing forzado:**

```
y'' + 0.2·y' + f(y) = 0.8·cos(1.2·t)
```

donde:
- `f(y) = 1.0·y + 0.5·y³` (función no lineal a identificar)
- Condiciones iniciales: `y₀ = 0.5`, `v₀ = 0.0`

## 💡 Recomendaciones Prácticas

### USAR REGRESOR HOMOTÓPICO cuando:

✅ Datos **muy escasos** (N ≤ 20)
✅ Paso temporal **grande** (T > 0.5 s)
✅ Método tradicional **falla** (error > 1.0)
✅ Precisión crítica con **pocos datos**
✅ **Mejora típica**: +50% a +99%

### USAR MÉTODO TRADICIONAL cuando:

✅ Datos **moderados/abundantes** (N ≥ 30)
✅ Paso temporal **pequeño** (T < 0.5 s)
✅ **Velocidad** es importante (1000× más rápido)
✅ Método regresor no justifica el costo

## 📚 Contexto del Proyecto

Este es el quinto trabajo de una serie sobre identificación de sistemas con RBF:

1. **PublicationA**: Método tradicional (despeje + RBF)
2. **PublicationB**: Optimización directa (sin derivadas)
3. **PublicationC**: Validación en péndulo con fricción
4. **PublicationD**: Oscilador de Duffing (límites de optimización directa)
5. **PublicationE**: **Regresor homotópico con RBF** ← Este trabajo

## 🔑 Conclusiones Principales

1. **No existe un método universalmente superior**
   - La elección depende de: cantidad de datos, paso temporal, complejidad del sistema

2. **El regresor rescata casos difíciles**
   - Con N ≤ 20, mejoras de hasta +99.9%
   - Alternativa viable cuando la optimización directa falla

3. **Punto de transición bien definido**
   - N ≈ 20-25 marca el cambio entre regímenes

4. **Eficiencia computacional**
   - 0.31 s para 3000 puntos
   - Comparable a RK45 pero usando directamente la RBF

## 📧 Información del Proyecto

- **Proyecto**: Identificación de Sistemas con RBF
- **Publicación**: E
- **Fecha**: Marzo 2026
- **Sistema**: Oscilador de Duffing Forzado
- **Método**: Regresor Homotópico con RBF Analítica

---

**Mensaje final**: El regresor homotópico es particularmente valioso cuando los datos son escasos y la optimización directa no es viable. Cada método tiene su lugar óptimo en el espectro de aplicaciones.
