# Contratos de Ajuste de Coeficientes de Mezcla mᵢ
# Proyecto: homotopy_regressors + MBP
# Autor: Rodolfo H. Rodrigo - UNSJ
# Fecha: 25/02/2026

## Contexto

Ecuación a resolver:

    ŷ(k+1) = regresor(yp(k), yp(k-1), u(k)) + Σ mᵢ · Nᵢ(yp(k), yp(k-1))

Donde:
- regresor: resuelve la parte conocida de la ODE (diferencias finitas + homotopía)
- Nᵢ: grupo i de la MBP, pre-entrenado con LM, pesos CONGELADOS
- mᵢ: coeficientes de mezcla a ajustar (únicos parámetros libres)
- N grupos, cada uno aproxima una componente canónica de la dinámica

Error en cada muestra k:

    e(k) = yp(k) - ŷ(k)

Derivadas disponibles (analíticas, de la red):
- Nᵢ(y): salida del grupo i
- Nᵢ'(y): dNᵢ/dy  (de la cadena de sigmoides)
- Nᵢ''(y): d²Nᵢ/dy²
- Nᵢ'''(y): d³Nᵢ/dy³

g'(yₖ) = coeff_derivada/T + Σ mᵢ · Nᵢ'(yₖ)

---

## Contrato A: Búsqueda Exhaustiva (Baseline)

### Nombre
`fit_exhaustive`

### Entrada
- `grupos`: lista de N funciones [N₁, N₂, ..., Nₙ], cada una callable(y) → float
- `y_medido`: array de M muestras medidas
- `y_regresor`: array de M muestras del regresor (parte conocida)
- `u`: array de excitación

### Salida
- `m_opt`: array de N coeficientes óptimos
- `combo_opt`: máscara booleana de grupos activos
- `error_opt`: MSE mínimo

### Algoritmo
```
Para cada subconjunto S de {1,...,N} (2ᴺ-1 combinaciones):
    mᵢ = 1/|S| para i ∈ S, 0 para i ∉ S
    Opcionalmente: refinar mᵢ minimizando MSE (least squares lineal)
    Calcular MSE sobre todas las muestras
Devolver S con menor MSE
```

### Complejidad
- Evaluaciones: (2ᴺ-1) × M
- Factible para N ≤ 15
- Sin derivadas, sin gradiente

### Ventajas
- Garantía de óptimo global
- Sin hiperparámetros
- Sin problemas de convergencia

### Desventajas
- Exponencial en N
- No es online
- Requiere todas las muestras

---

## Contrato B: LMS Muestra a Muestra (z₁ en mᵢ)

### Nombre
`fit_lms`

### Entrada
- `grupos`: lista de N funciones [N₁, N₂, ..., Nₙ]
- `y_medido`: array de M muestras
- `u`: array de excitación
- `y0, y1`: condiciones iniciales
- `T`: período de muestreo
- `eta`: paso de aprendizaje (default 0.01)
- `epochs`: número de pasadas sobre los datos (default 100)

### Salida
- `m`: array de N coeficientes ajustados
- `error_history`: MSE por época
- `y_pred`: predicción final

### Algoritmo
```
m = [1/N, 1/N, ..., 1/N]  # inicialización uniforme

Para cada época:
    Para k = 2, ..., M-1:
        # Forward pass: resolver ODE con regresor homotópico usando m actual
        ŷ(k) = resolver_paso(y, m, grupos, u, k, T)
        
        # Error
        e(k) = yp(k) - ŷ(k)
        
        # Gradiente (z₁ en dirección mᵢ)
        Para cada grupo i:
            δmᵢ = eta · e(k) · Nᵢ(yp(k)) / g'(yp(k))
            mᵢ = mᵢ + δmᵢ
```

### Complejidad
- Evaluaciones por época: M × N
- Total: epochs × M × N
- Online: se puede cortar en cualquier momento

### Ventajas
- Simple
- Online (muestra a muestra)
- Bajo costo por paso
- Apto para microcontrolador

### Desventajas
- Convergencia lenta
- Sensible a eta
- Puede oscilar

### Nota teórica
Es exactamente la regla delta (Widrow-Hoff) aplicada a los mᵢ.
El término Nᵢ(yp(k))/g'(yp(k)) es el gradiente del regresor
respecto a mᵢ, calculado analíticamente.

---

## Contrato C: Homotópico Muestra a Muestra (z₁ + z₂ en mᵢ)

### Nombre
`fit_homotopic`

### Entrada
- `grupos`: lista de N funciones [N₁, N₂, ..., Nₙ]
- `grupos_d1`: lista de N funciones [N₁', N₂', ..., Nₙ']  (primeras derivadas)
- `grupos_d2`: lista de N funciones [N₁'', N₂'', ..., Nₙ''] (segundas derivadas)
- `y_medido`: array de M muestras
- `u`: array de excitación
- `y0, y1`: condiciones iniciales
- `T`: período de muestreo
- `epochs`: número de pasadas (default 50)

### Salida
- `m`: array de N coeficientes ajustados
- `error_history`: MSE por época
- `y_pred`: predicción final

### Algoritmo
```
m = [1/N, ..., 1/N]

Para cada época:
    Para k = 2, ..., M-1:
        # Forward pass
        ŷ(k) = resolver_paso(y, m, grupos, u, k, T)
        e(k) = yp(k) - ŷ(k)
        
        # Función de error respecto a mᵢ
        # E(m) = (yp(k) - ŷ(k))² / 2
        
        Para cada grupo i:
            # Derivadas de E respecto a mᵢ
            dE_dmi  = -e(k) · Nᵢ(yp(k)) / g'
            d2E_dmi = (Nᵢ(yp(k)) / g')² - e(k) · [Nᵢ'·g' - Nᵢ·g''] / g'³
            
            # z₁: Newton en mᵢ
            δmᵢ_1 = -dE_dmi / d2E_dmi
            
            # z₂: corrección Halley en mᵢ  
            # (requiere d³E/dmᵢ³, pero truncamos en z₁+z₂ del error)
            
            # Alternativa pragmática: z₁ de g, z₂ de g
            # aplicados en la dirección mᵢ
            
            # Actualizar
            mᵢ = mᵢ + δmᵢ_1
```

### Complejidad
- Evaluaciones por época: M × N × 3 (f, f', f'')
- Total: epochs × M × N × 3
- ~3× más caro que B por paso, pero converge en ~3× menos épocas

### Ventajas
- Convergencia cuadrática (Newton en mᵢ)
- Sin hiperparámetro eta
- Usa las derivadas que ya tenemos

### Desventajas
- Más cálculo por paso
- Newton puede diverger si d2E ≈ 0
- Necesita safeguard (limitar paso)

### Nota teórica
z₁ en mᵢ = regla delta / curvatura = Newton-Raphson en el espacio de mᵢ.
Es backpropagation de segundo orden, resuelto analíticamente.
Cada grupo se ajusta independientemente (diagonal de la Hessiana).

---

## Contrato D: Batch Levenberg-Marquardt

### Nombre
`fit_lm_batch`

### Entrada
- `grupos`: lista de N funciones
- `y_medido`: array de M muestras
- `u`: array de excitación
- `y0, y1`: condiciones iniciales
- `T`: período de muestreo
- `max_iter`: máximo de iteraciones LM (default 20)
- `lambda0`: damping inicial (default 0.01)

### Salida
- `m`: array de N coeficientes ajustados
- `error_history`: MSE por iteración
- `y_pred`: predicción final

### Algoritmo
```
m = [1/N, ..., 1/N]
lambda = lambda0

Para iter = 1, ..., max_iter:
    # Forward pass completo
    y_pred = resolver_ode_completa(m, grupos, u, y0, y1, T, M)
    e = y_medido - y_pred   # vector de errores (M × 1)
    
    # Jacobiano J (M × N)
    Para cada muestra k y cada grupo i:
        J[k, i] = -Nᵢ(yp(k)) / g'(yp(k))
    
    # Actualización LM
    H = JᵀJ + lambda · I        # (N × N)
    grad = Jᵀe                    # (N × 1)
    δm = solve(H, grad)           # Cholesky, (N × 1)
    
    # Evaluar nuevo m
    m_new = m + δm
    y_new = resolver_ode_completa(m_new, ...)
    mse_new = mean((y_medido - y_new)²)
    
    Si mse_new < mse_old:
        m = m_new
        lambda = lambda / 10
    Sino:
        lambda = lambda * 10
```

### Complejidad
- Por iteración: M × N (Jacobiano) + N³ (Cholesky) + M (forward pass)
- Total: max_iter × (M×N + N³)
- Para N=3, M=1000: trivial

### Ventajas
- Convergencia en 5-7 iteraciones (como los grupos)
- Robusto (damping λ previene divergencia)
- Óptimo para batch offline

### Desventajas
- No es online
- Requiere todas las muestras
- Jacobiano completo en memoria

### Nota teórica
Es exactamente el mismo LM que entrena los grupos, pero aplicado
a los N coeficientes de mezcla en vez de a los pesos de la red.
La dimensión del problema es N (típicamente 3-8), no 200 pesos.
El Jacobiano tiene una fila por muestra y una columna por grupo.
Sistema sobredeterminado: M >> N → solución robusta.

---

## Contrato E: Mini-Batch Homotópico

### Nombre
`fit_minibatch_homotopic`

### Entrada
- `grupos`: lista de N funciones
- `grupos_d1, grupos_d2`: derivadas
- `y_medido`: array de M muestras
- `u`: array de excitación
- `y0, y1`: condiciones iniciales
- `T`: período de muestreo
- `batch_size`: tamaño del mini-batch (default 50)
- `epochs`: pasadas completas (default 10)

### Salida
- `m`: array de N coeficientes ajustados
- `error_history`: MSE por mini-batch
- `y_pred`: predicción final

### Algoritmo
```
m = [1/N, ..., 1/N]

Para cada época:
    Para cada bloque de batch_size muestras [k_start, k_end]:
        # Acumular gradiente y curvatura sobre el bloque
        grad_acum = zeros(N)
        curv_acum = zeros(N)
        
        Para k en [k_start, k_end]:
            ŷ(k) = resolver_paso(y, m, grupos, u, k, T)
            e(k) = yp(k) - ŷ(k)
            
            Para cada grupo i:
                gi = Nᵢ(yp(k)) / g'(yp(k))
                grad_acum[i] += e(k) · gi           # z₁ acumulado
                curv_acum[i] += gi² - e(k) · (...)   # z₂ acumulado
        
        # Corrección homotópica sobre el bloque
        Para cada grupo i:
            # z₁ promediado
            δmᵢ_1 = grad_acum[i] / batch_size
            
            # z₂ promediado (corrección de curvatura)
            δmᵢ_2 = ... función de curv_acum ...
            
            mᵢ = mᵢ + δmᵢ_1 + δmᵢ_2
```

### Complejidad
- Por mini-batch: batch_size × N × 3
- Total: epochs × (M/batch_size) × batch_size × N × 3 = epochs × M × N × 3
- Similar a C pero con mejor dirección de descenso

### Ventajas
- Más estable que C (promedia sobre varias muestras)
- Más rápido que D (no necesita todas las muestras)
- Semi-online (puede actualizar cada batch_size muestras)
- z₂ sobre el bloque da mejor estimación de curvatura

### Desventajas
- Hiperparámetro batch_size
- No tan óptimo como D
- Más complejo de implementar

### Nota teórica
Es el híbrido natural entre C (muestra a muestra) y D (batch completo).
El z₁ acumulado sobre el bloque es equivalente al gradiente medio.
El z₂ acumulado es equivalente a la diagonal de la Hessiana media.
Converge como Newton pero con estimación estadística de la curvatura.

---

## Resumen Comparativo

| Contrato | Método | Online | Derivadas | Convergencia | Hiperparámetros | Micro |
|----------|--------|--------|-----------|--------------|-----------------|-------|
| A | Exhaustivo | No | No | Garantizada | Ninguno | No |
| B | LMS (z₁) | Sí | No | Lenta | eta | Sí |
| C | Homotópico (z₁+z₂) | Sí | Sí (N',N'') | Cuadrática | Ninguno | Sí |
| D | LM Batch | No | Sí (N') | 5-7 iter | lambda0 | No |
| E | Mini-batch homo | Semi | Sí (N',N'') | Rápida | batch_size | Semi |

## Protocolo de Evaluación (mañana)

### Planta de prueba
Narendra Ejemplo 2: y(k+1) = f[y(k), y(k-1)] + u(k)
con f desconocida, aproximada por 3 grupos MBP.

### Métricas
1. MSE final
2. Número de evaluaciones de f hasta convergencia
3. Estabilidad (¿diverge?)
4. Sensibilidad a condiciones iniciales de mᵢ
5. Rendimiento con M = {5, 10, 20, 50, 100} muestras

### Test adicional
- Ruido en y_medido (SNR = 20dB, 10dB)
- Grupo espurio (N=4 con 3 activos)
- Cambio abrupto de mᵢ en t = T/2 (tracking)

---

## Conexión Teórica

Narendra (1990): N caja negra, backprop clásico, 100000 pasos, sin derivadas.

Rodrigo (2026): N con derivadas analíticas (red diferenciable), serie homotópica
resuelve tanto el forward pass (predicción) como el backward pass (ajuste de mᵢ).

El regresor homotópico unifica:
- Forward: y(k+1) = resolver g(y) = 0 con z₁+z₂ en dirección y
- Backward: δmᵢ = resolver ∂E/∂mᵢ = 0 con z₁+z₂ en dirección mᵢ

Mismo motor. Dos direcciones. Sin backpropagation iterativo.
