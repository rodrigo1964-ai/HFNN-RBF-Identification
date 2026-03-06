# Regresor Homotópico con RBF para Identificación de Sistemas No Lineales

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📄 Descripción General

Este repositorio contiene la implementación completa de **4 casos de estudio** que demuestran el uso del **regresor homotópico de 3 puntos** (serie de Liao) para resolver ecuaciones diferenciales no lineales, incluyendo la innovadora combinación con **Redes de Base Radial (RBF)** para identificación de funciones desconocidas.

**Documento principal:** `main_revised.pdf` - Análisis completo de los 4 casos de estudio

## 📁 Repository Structure

```
HFNN-RBF-Identification/
├── CaseStudy_1/          # Case 1: Polynomial nonlinear equation
├── CaseStudy_2/          # Case 2: Trigonometric nonlinear equation
├── CaseStudy_3/          # Case 3: First-order RBF identification
├── CaseStudy_4/          # Case 4: Second-order Duffing oscillator with RBF
├── docs/                  # Documentation and papers
│   ├── CaseStudy_*/      # Documentation for each case study
│   └── main_revised.pdf  # Complete analysis document
├── CITATION.cff          # Citation information with DOI
├── LICENSE               # GPL-3.0 license
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## 🎯 Objetivos del Proyecto

1. **Validar el regresor homotópico** para diferentes tipos de ecuaciones diferenciales
2. **Integrar RBF** para aproximar funciones no lineales desconocidas
3. **Comparar precisión** con métodos tradicionales (RK4, odeint)
4. **Demostrar aplicabilidad** en identificación de sistemas reales

## 📚 Casos de Estudio

### Caso 1: Ecuación No Lineal Polinomial
**Carpeta:** [`CaseStudy_1/`](CaseStudy_1/)

**Ecuación:** `y' + y² = sin(5t)`

**Características:**
- ✅ Función no lineal polinomial simple
- ✅ Derivadas analíticas explícitas
- ✅ Error RMS ~10⁻⁴ (N=500)
- ✅ Tiempo: ~3-5 ms

**Archivos principales:**
- `caso1_regressor.py` - Implementación del regresor
- `test_caso1.py` - Tests y análisis de convergencia
- `generate_figures.py` - Generación de figuras
- `README.md` - Documentación completa

**Conclusión:** El regresor homotópico alcanza excelente precisión para sistemas no lineales de primer orden.

---

### Caso 2: Ecuación No Lineal Trigonométrica
**Carpeta:** [`CaseStudy_2/`](CaseStudy_2/)

**Ecuación:** `y' + sin²(y) = sin(5t)`

**Características:**
- ✅ Función no lineal trigonométrica
- ✅ Derivadas trigonométricas: f'(y) = sin(2y)
- ✅ Error RMS ~10⁻³ (N=500)
- ✅ Estabilidad numérica por acotación de funciones trigonométricas

**Archivos principales:**
- `caso2_regressor.py` - Implementación del regresor
- `test_caso2.py` - Tests y análisis
- `generate_figures.py` - Figuras
- `README.md` - Documentación

**Conclusión:** El método mantiene precisión con funciones trigonométricas, aunque requiere mayor cuidado por la periodicidad.

---

### Caso 3: Identificación con RBF (Primer Orden)
**Carpeta:** [`CaseStudy_3/`](CaseStudy_3/)

**Ecuación:** `y' + β(y) = sin(5t)` donde `β(y) = 0.1y³ + 0.1y² + y - 1` es **desconocida**

**Características:**
- 🌟 **Primera introducción de RBF** para funciones desconocidas
- ✅ RBF Gaussiana con integral: ∫exp(-(x-c)²/σ²)dx
- ✅ Optimización con Nelder-Mead
- ✅ Error RMS ~10⁻³ (N=100, k=5 neuronas)
- ✅ Regresor de Liao con derivadas de RBF

**Archivos principales:**
- `rbf_integration.py` - Módulo de RBF con integrales
- `caso3_regressor_rbf.py` - Regresor con RBF
- `optimize_rbf_caso3.py` - Optimización Nelder-Mead
- `test_caso3.py` - Suite de tests
- `generate_figures.py` - Figuras
- `README.md` - Documentación

**Conclusión:** La combinación RBF + regresor homotópico permite identificar funciones desconocidas con alta precisión.

---

### Caso 4: Oscilador de Duffing con RBF
**Carpeta:** [`CaseStudy_4/`](CaseStudy_4/)

**Ecuación:** `y'' + 0.2·y' + f(y) = 0.8·cos(1.2·t)` donde `f(y) = 1.0·y + 0.5·y³`

**Características:**
- 🌟 **Sistema de segundo orden muy no lineal**
- ✅ Regresor homotópico de orden superior
- ✅ RBF con derivadas analíticas hasta orden 3
- ✅ Error RMS ~4.09×10⁻² (N=3000)
- ✅ Análisis completo de sensibilidad y regímenes óptimos

**Archivos principales:**
- `rbf_analytical.py` - RBF con derivadas analíticas
- `duffing_regressor_rbf.py` - Regresor principal
- `optimize_rbf_regressor.py` - Comparación métodos
- `sensitivity_analysis_regressor.py` - Análisis de sensibilidad
- `test_duffing_regressor.py` - Tests
- `generate_figures.py` - Figuras
- `duffing_regressor_paper.pdf` - Paper completo (8 páginas)
- `README.md` - Documentación

**Conclusión:** El regresor homotópico es especialmente valioso con datos escasos (N≤20), logrando mejoras de +50% a +99% sobre métodos tradicionales.

---

## 🔬 Método del Regresor Homotópico

### Principio Fundamental

El regresor homotópico de 3 puntos resuelve ecuaciones diferenciales mediante:

1. **Discretización** usando diferencias finitas hacia atrás (3 puntos)
2. **Tres correcciones iterativas** por punto (serie de Liao):
   - **z1**: Corrección Newton-Raphson
   - **z2**: Corrección de orden 2 (término cuadrático)
   - **z3**: Corrección de orden 3 (término cúbico)

### Ecuaciones de Primer Orden

Para `y' + f(y) = u(t)`:

```
Discretización:
  y'[k] ≈ (3y[k] - 4y[k-1] + y[k-2]) / (2T)

Función residual:
  g(y[k]) = (3/2)·y[k]/T - 2·y[k-1]/T + (1/2)·y[k-2]/T + f(y[k]) - u[k]

Correcciones:
  z1: y[k] ← y[k] - g/g'
  z2: y[k] ← y[k] - (1/2)·g²·g''/(g')³
  z3: y[k] ← y[k] - (1/6)·g³·(-g'''·g' + 3·g''²)/(g')⁵
```

donde `g' = 3/(2T) + f'(y[k])`, `g'' = f''(y[k])`, `g''' = f'''(y[k])`

### Integración con RBF

Para funciones desconocidas `β(y)`, se aproxima con RBF Gaussiana:

```
β(y) ≈ Σ wⱼ·φⱼ(y)   donde   φⱼ(y) = exp(-(y-cⱼ)²/(2σ²))
```

El regresor requiere:
- `β(y)` ← Integral de RBF
- `β'(y)` ← RBF
- `β''(y)` ← Derivada de RBF
- `β'''(y)` ← Segunda derivada de RBF

## 📊 Resultados Comparativos

| Caso | Ecuación | N Puntos | Error RMS | Tiempo | RBF |
|------|----------|----------|-----------|--------|-----|
| 1 | y' + y² = sin(5t) | 500 | ~10⁻⁴ | 3-5 ms | No |
| 2 | y' + sin²(y) = sin(5t) | 500 | ~10⁻³ | 6 ms | No |
| 3 | y' + β(y) = sin(5t) | 100 | ~10⁻³ | 15 ms | Sí (k=5) |
| 4 | Duffing (2° orden) | 3000 | 4.09×10⁻² | 309 ms | Sí (k=?) |

### Orden de Convergencia

- **Casos 1-3:** O(T^0.9-1.0) - Reducir T a la mitad mejora error ~2×
- **Caso 4:** Más complejo por no linealidad fuerte

## 🚀 Cómo Usar Este Repositorio

### Requisitos

**Python 3.8 o superior**

Instalar dependencias:

```bash
pip install -r requirements.txt
```

O manualmente:

```bash
pip install numpy scipy matplotlib sympy
```

### Ejecutar un Caso de Estudio

**Ejemplo: Caso 1**

```bash
cd CaseStudy_1

# Test principal
python3 caso1_regressor.py

# Tests completos
python3 test_caso1.py

# Generar figuras
python3 generate_figures.py
```

**Repite para otros casos** (CaseStudy_2, CaseStudy_3, CaseStudy_4)

### Estructura de Cada Caso

Todos los casos siguen una estructura consistente:

```
CaseStudy_X/
├── casoX_regressor[_rbf].py    # Implementación principal
├── test_casoX.py               # Suite de tests
├── generate_figures.py         # Generación de figuras
├── README.md                   # Documentación del caso
├── [rbf_*.py]                  # Módulos RBF (casos 3 y 4)
├── [optimize_*.py]             # Optimización (casos 3 y 4)
└── [figuras generadas]         # PNG de alta calidad
```

## 📖 Publicaciones y Referencia

Este proyecto forma parte de una serie de trabajos sobre identificación de sistemas:

1. **PublicationA**: Método tradicional (despeje + RBF)
2. **PublicationB**: Optimización directa (sin derivadas)
3. **PublicationC**: Validación en péndulo con fricción
4. **PublicationD**: Oscilador de Duffing (límites de optimización)
5. **PublicationE**: Regresor homotópico con RBF ← **Este trabajo**

**Documento principal:** Leer `main_revised.pdf` para el análisis completo de los 4 casos.

## 💡 Recomendaciones Prácticas

### Cuándo Usar el Regresor Homotópico

**✅ RECOMENDADO:**
- Sistemas lineales o moderadamente no lineales
- Datos escasos (N ≤ 20-30)
- Aplicaciones en tiempo real o embebidos
- Derivadas analíticas calculables
- Simplicidad de implementación importante

**⚠️ CONSIDERAR ALTERNATIVAS:**
- Sistemas muy no lineales o caóticos (requiere T muy pequeño)
- Derivadas difíciles de calcular
- Máxima precisión con mínimo ajuste requerido

### Cuándo Usar RBF + Regresor

**✅ USAR RBF cuando:**
- Función no lineal `f(y)` o `β(y)` es **desconocida**
- Solo se tienen datos de la función
- Precisión aceptable con k=5-10 neuronas
- Se acepta costo de optimización inicial

**⚠️ EVITAR RBF cuando:**
- Función analítica conocida (usar derivadas directas)
- Muy pocos datos para entrenar (k<3)
- Tiempo de entrenamiento es crítico

## 📂 Archivos Adicionales

### Código Base Reutilizable

- `ode_rbf_solver.py` - Solver genérico de ODEs con RBF
- `ode_rbf_approximation.py` - Aproximación de ODEs
- `ode_rbf_identification.py` - Identificación completa
- `ode_rbf_data_requirements.py` - Análisis de requisitos de datos
- `rbf_*.py` - Diferentes implementaciones de RBF

### Notebooks Jupyter (Originales)

Cada carpeta `CaseStudy_X/` contiene los notebooks Jupyter originales del desarrollo:
- `Caso_X_2p_v1.ipynb` - Versión principal
- `Caso_X_2p.ipynb` - Versión preliminar
- Archivos PDF de exportación

Estos notebooks son referencias históricas; **usar los scripts Python** para reproducibilidad.

## 🔑 Conclusiones Principales

1. **El regresor homotópico es competitivo** con RK4/odeint para sistemas de complejidad baja a moderada

2. **La integración con RBF** permite identificar funciones desconocidas, abriendo aplicaciones en sistemas reales

3. **Ventaja en datos escasos**: Con N≤20, el regresor supera métodos tradicionales (+50% a +99%)

4. **Orden de convergencia O(T)**: Predictible y escalable, aunque más lento que RK4 (O(T⁴))

5. **Implementación simple**: Solo 3 correcciones por paso, ideal para embebidos

6. **No hay método universalmente superior**: La elección depende de:
   - Cantidad de datos disponibles
   - Grado de no linealidad
   - Requisitos de precisión vs velocidad
   - Disponibilidad de derivadas analíticas

## 📧 Información del Proyecto

- **Autor**: Rodolfo H. Rodrigo
- **Institución**: Universidad Nacional de San Juan (UNSJ)
- **Fecha**: Marzo 2026
- **Métodos**: Regresor Homotópico (serie de Liao) + RBF Gaussiana

## 📝 Cómo Citar

Si usas este código en tu investigación, por favor cita:

```bibtex
@software{rodrigo2026hfnn_rbf,
  author = {Rodrigo, Rodolfo H.},
  title = {Regresor Homotópico con RBF para Identificación de Sistemas No Lineales},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/rodrigo1964-ai/HFNN-RBF-Identification}
}
```

## 🤝 Contribuciones

Este es un proyecto académico de investigación. Para preguntas o colaboraciones, contactar a través de GitHub Issues.

## 📜 Licencia

MIT License - Ver archivo `LICENSE` para detalles.

---

**Próximos Pasos Sugeridos:**
1. Implementar control adaptativo del paso temporal T
2. Extender a sistemas MIMO (múltiples entradas/salidas)
3. Portar a microcontroladores (ARM, ESP32)
4. Comparar con métodos adaptativos (RK45, Dormand-Prince)
5. Aplicar a sistemas físicos reales (robótica, control de procesos)

---

**⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!**
