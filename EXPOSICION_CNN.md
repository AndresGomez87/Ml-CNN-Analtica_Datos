# Redes Neuronales Convolucionales (CNN)
## Guía completa para la exposición — Analítica de Datos

> **Universidad de Antioquia · Instituto de Matemáticas**  
> Tópicos de Machine Learning · Profesor: Duván Cataño  
> Este documento es el material de referencia completo para la exposición.  
> Cubre la teoría, intuición, matemática y ejemplo práctico.

---

## Tabla de contenidos

1. [Introducción al problema](#1-introducción-al-problema)
2. [¿Por qué no funcionan las redes tradicionales?](#2-por-qué-no-funcionan-las-redes-tradicionales)
3. [Evolución histórica](#3-evolución-histórica)
4. [Fundamento teórico de la convolución](#4-fundamento-teórico-de-la-convolución)
5. [Función de activación ReLU](#5-función-de-activación-relu)
6. [Capa de Pooling](#6-capa-de-pooling)
7. [Capas densas y clasificación final](#7-capas-densas-y-clasificación-final)
8. [Arquitectura completa](#8-arquitectura-completa)
9. [Entrenamiento: Backpropagation en CNNs](#9-entrenamiento-backpropagation-en-cnns)
10. [Transfer Learning](#10-transfer-learning)
11. [Caso práctico: COVID-19 X-Ray](#11-caso-práctico-covid-19-x-ray)
12. [Casos de uso reales](#12-casos-de-uso-reales)
13. [Ventajas y limitaciones](#13-ventajas-y-limitaciones)
14. [El futuro: más allá de las CNNs](#14-el-futuro-más-allá-de-las-cnns)
15. [Ideas pedagógicas para la exposición](#15-ideas-pedagógicas-para-la-exposición)
16. [Referencias](#16-referencias)

---

## 1. Introducción al problema

### ¿Cuál es el problema central?

Las computadoras reciben imágenes como **matrices de números**: una imagen RGB de 224×224 píxeles es simplemente un arreglo tridimensional de forma `(224, 224, 3)`, donde cada posición contiene un valor entre 0 y 255 que representa la intensidad de los canales Rojo, Verde y Azul.

El desafío: ¿cómo enseñar a un algoritmo a **entender** lo que hay en esa matriz de números?

```
Imagen de un pulmón sano:
[[120, 118, 115, ...],   ← Fila 0 de píxeles
 [119, 117, 114, ...],   ← Fila 1 de píxeles
 ...                     ← 224 filas en total
]
```

Para una persona, reconocer si un pulmón tiene neumonía requiere años de entrenamiento médico. Para una red neuronal, el desafío es **extraer patrones relevantes** (opacidades, consolidaciones, bordes pulmonares) directamente de esos miles de números.

### ¿Por qué es importante resolverlo?

- **Diagnóstico médico**: Radiólogos leen entre 50-100 estudios por día. Una CNN puede analizar miles en el mismo tiempo.
- **Vehículos autónomos**: Un auto necesita reconocer peatones y señales en milisegundos.
- **Seguridad**: Face ID de Apple mapea 30,000 puntos del rostro en tiempo real.
- **Ciencia**: Clasificar millones de galaxias que ningún humano podría revisar manualmente.

> **Idea para la exposición**: Empezar mostrando una radiografía de tórax y preguntarle al público "¿ven algo anormal aquí?". Luego mostrar la misma imagen con las predicciones de la CNN superpuestas. El contraste es impactante.

---

## 2. ¿Por qué no funcionan las redes tradicionales?

### El Perceptrón Multicapa (MLP) y sus limitaciones con imágenes

Un MLP es una red donde **cada neurona está conectada con todas las neuronas de la capa anterior** (fully connected). Para texto o datos tabulares funciona bien. Para imágenes, tiene tres problemas fundamentales.

### Problema 1: La pesadilla computacional

Considera una imagen modesta de **256×256 píxeles a color (RGB)**:

```
Neuronas de entrada = 256 × 256 × 3 = 196,608 neuronas
```

Si la primera capa oculta tiene solo 1,000 neuronas:

```
Parámetros = 196,608 × 1,000 = 196,608,000 ≈ 200 millones de pesos
```

Solo en la primera capa. Esto es:
- **Computacionalmente inmanejable** para entrenar (memoria y tiempo)
- **Imposible de generalizar** con pocos datos (más parámetros que ejemplos de entrenamiento)
- **Extremadamente propenso al overfitting**

> **Intuición**: Es como intentar memorizar un libro completo en lugar de aprender a leer. Con demasiados parámetros, el modelo memoriza los datos de entrenamiento en lugar de aprender patrones generales.

### Problema 2: Pérdida de estructura espacial

El MLP requiere **aplanar (flatten)** la imagen en un vector 1D antes de procesarla:

```
Imagen 2D (5×5):          Vector 1D (después de flatten):
┌───┬───┬───┬───┬───┐
│ A │ B │ C │ D │ E │     [A, B, C, D, E, F, G, H, I, J, ...]
├───┼───┼───┼───┼───┤
│ F │ G │ H │ I │ J │
└───┴───┴───┴───┴───┘
```

Al aplanar, **se pierde toda información sobre qué píxeles son vecinos**. El modelo no sabe que A está al lado de B, o que F está debajo de A.

Consecuencias:
- **Destrucción topológica**: Se pierde la geometría de la imagen
- **Sin invariancia a traslación**: Un gato en la esquina superior izquierda y un gato en la esquina inferior derecha se ven como patrones completamente distintos para un MLP
- **Incapacidad para detectar bordes**: Un borde es una relación entre píxeles vecinos — relación que el flatten destruye

> **Intuición**: Es como intentar reconocer una melodía si te dan las notas en orden aleatorio. Las notas pueden ser las mismas, pero sin el orden temporal (o espacial en el caso de imágenes) la información esencial se pierde.

### Problema 3: Sin compartición de parámetros

En un MLP, los pesos que detectan "un ojo" en la esquina superior izquierda son **completamente diferentes** de los que detectarían "un ojo" en la esquina inferior derecha. El modelo debe aprender a detectar el mismo patrón en cada posición posible por separado.

Las CNNs resuelven esto con **filtros convolucionales que se aplican en toda la imagen**.

---

## 3. Evolución histórica

### Línea del tiempo de las CNNs

#### 1959 — El origen biológico: Hubel y Wiesel

David Hubel y Torsten Wiesel ganaron el Premio Nobel de Fisiología (1981) por descubrir cómo funciona la **corteza visual** de los mamíferos:

- Las neuronas de la corteza visual no responden a toda la imagen
- Responden solo a **estímulos locales**: un borde, una orientación específica, en una región limitada del campo visual
- Las neuronas se organizan **jerárquicamente**: células simples detectan bordes básicos, células complejas combinan esos bordes para detectar formas

Esta jerarquía biológica inspiró directamente la arquitectura de las CNNs:

```
Biología:          CNN equivalente:
Células simples  → Primeras capas convolucionales (bordes)
Células complejas → Capas más profundas (formas, objetos)
Corteza IT       → Capas densas (clasificación)
```

#### 1980 — Neocognitrón (Fukushima)

Kunihiko Fukushima propone el **Neocognitrón**: el primer modelo computacional inspirado en la jerarquía de Hubel y Wiesel. Introduce la idea alternante de capas de detección (S-cells) y submuestreo (C-cells) — lo que hoy son Conv y Pool.

No usaba backpropagation y tenía limitaciones, pero estableció los principios fundamentales.

#### 1998 — LeNet-5 (Yann LeCun)

Yann LeCun combina el Neocognitrón con backpropagation y crea **LeNet-5**: la primera CNN exitosa para producción real.

- **Aplicación**: Reconocimiento de dígitos escritos a mano en cheques bancarios para AT&T
- **Arquitectura**: Conv(6 filtros) → AvgPool → Conv(16 filtros) → AvgPool → Dense(120) → Dense(84) → Softmax
- **Resultado**: 99.2% de precisión en MNIST

LeNet-5 demostró que las CNNs funcionan en la práctica. El problema: no había suficiente poder computacional ni datos para escalar.

#### 2012 — AlexNet y la revolución del Deep Learning

Alex Krizhevsky, Ilya Sutskever y Geoffrey Hinton presentan **AlexNet** en el concurso ImageNet (ILSVRC 2012):

- **Error de clasificación**: 15.3% vs 26.2% del segundo lugar (humano-comparable: ~5%)
- **Innovaciones clave**:
  - Uso de **GPUs** para entrenamiento (NVIDIA GTX 580)
  - **ReLU** en lugar de tanh/sigmoid → entrenamiento 6× más rápido
  - **Dropout** para regularización
  - **Data augmentation** para aumentar el tamaño efectivo del dataset
- **Arquitectura**: 5 capas conv + 3 capas densas, ~60M parámetros

AlexNet marcó el inicio de la **era moderna del Deep Learning**.

#### 2014 — VGGNet y GoogLeNet

- **VGGNet** (Oxford): demuestra que la profundidad importa. Usa solo filtros 3×3 pero apila hasta 19 capas. Simple pero efectiva.
- **GoogLeNet/Inception**: introduce los módulos Inception para hacer las redes más anchas y profundas eficientemente.

#### 2015 — ResNet (Redes Residuales)

Microsoft Research presenta **ResNet**: 152 capas con **conexiones residuales (skip connections)**:

```
Salida = F(x) + x    ← La red aprende la "residual", no la transformación completa
```

Sin las skip connections, redes muy profundas sufren **vanishing gradient** (los gradientes se vuelven tan pequeños que los pesos no se actualizan). ResNet resuelve esto permitiendo que el gradiente fluya directamente a través de las conexiones de atajo.

**Resultado**: Supera a humanos en ImageNet con 3.57% de error top-5.

#### 2017 en adelante — EfficientNet, MobileNet, Vision Transformers

- **MobileNet (Google, 2017)**: Diseñada para dispositivos móviles. Usa convoluciones separables en profundidad para reducir parámetros 8-9× con poca pérdida de precisión.
- **EfficientNet (2019)**: Escala ancho, profundidad y resolución simultáneamente usando NAS (Neural Architecture Search).
- **Vision Transformer (2020)**: Elimina la convolución completamente, tratando parches de imagen como tokens de un Transformer.

---

## 4. Fundamento teórico de la convolución

### ¿Qué es una convolución matemáticamente?

La operación de convolución discreta 2D entre una imagen `I` y un kernel `W` se define como:

```
S(i, j) = (I ★ W)(i, j) = Σ_m Σ_n I(i+m, j+n) · W(m, n)
```

Donde:
- `I` es la imagen de entrada (matriz 2D de píxeles)
- `W` es el kernel o filtro (matriz pequeña, típicamente 3×3 o 5×5)
- `S` es el mapa de características resultante (feature map)
- `(i, j)` es la posición actual del filtro
- `m, n` recorren las dimensiones del kernel

### Intuición: el filtro como "detector de patrón"

Piensa en el kernel como una **plantilla** que buscas en la imagen. La operación de convolución mide "cuánto se parece" cada parche de la imagen a esa plantilla.

**Ejemplo con un detector de bordes horizontales:**

```
Kernel (detector de bordes horizontales):
┌────┬────┬────┐
│ -1 │ -1 │ -1 │   ← pesos negativos arriba
├────┼────┼────┤
│  0 │  0 │  0 │   ← pesos cero en el medio
├────┼────┼────┤
│ +1 │ +1 │ +1 │   ← pesos positivos abajo
└────┴────┴────┘
```

Cuando este kernel se aplica sobre una región donde los píxeles de abajo son más brillantes que los de arriba → resultado grande positivo (borde horizontal detectado).

Cuando se aplica sobre una región uniforme → resultado ≈ 0 (no hay borde).

**Ejemplo numérico:**

```
Parche de imagen (región oscura arriba, clara abajo):
┌────┬────┬────┐
│  20│  25│  22│   ← píxeles oscuros (arriba)
├────┼────┼────┤
│  18│  20│  19│
├────┼────┼────┤
│ 200│ 210│ 205│   ← píxeles claros (abajo = borde)
└────┴────┴────┘

Convolución:
(-1)·20 + (-1)·25 + (-1)·22 +
  0·18  +   0·20  +   0·19  +
(+1)·200+ (+1)·210+ (+1)·205 =

= -67 + 0 + 615 = 548 ← valor alto → borde detectado!
```

### Parámetros de la convolución

#### Stride (paso)
El stride define cuántos píxeles se mueve el kernel en cada paso.

```
Stride = 1: el kernel se mueve de 1 en 1 → salida ≈ mismo tamaño que entrada
Stride = 2: el kernel se mueve de 2 en 2 → salida ≈ mitad del tamaño

Con imagen 6×6 y kernel 3×3:
  Stride 1 → salida 4×4
  Stride 2 → salida 2×2
```

#### Padding
El padding añade filas/columnas de ceros alrededor de la imagen.

```
Sin padding (valid): la imagen se "encoge" con cada convolución
Con padding (same):  la salida tiene el mismo tamaño que la entrada

Tamaño de salida (sin padding) = (N - F) / S + 1
  N = tamaño de entrada
  F = tamaño del filtro
  S = stride
```

#### Número de filtros y profundidad

En la práctica, no aplicamos un solo kernel sino **múltiples filtros en paralelo**:

```
Entrada: (H, W, C_in)    ← H altura, W ancho, C_in canales de entrada
Filtros: F kernels de (k × k × C_in)
Salida:  (H', W', F)     ← F mapas de características en la salida
```

Cada filtro aprende a detectar un patrón diferente. Con 32 filtros 3×3, la capa aprende 32 "detectores de patrón" diferentes.

### ¿Cómo aprende la red qué filtros usar?

Durante el entrenamiento, los **pesos de los kernels se ajustan automáticamente** mediante backpropagation para detectar los patrones que mejor minimicen el error.

No diseñamos los filtros manualmente. La red descubre sola que necesita detectores de bordes, detectores de texturas, detectores de formas, etc.

> **Intuición pedagógica**: Es como si contratáramos a alguien para diseñar sellos personalizados que, al estamparse sobre diferentes partes de la imagen, nos digan qué hay ahí. Pero en lugar de diseñarlos nosotros, el "sello" se modifica automáticamente cada vez que comete un error hasta que detecta lo que necesita.

### Jerarquía de características aprendidas

Las CNNs aprenden características de complejidad creciente a medida que se profundiza en la red:

```
Capa 1:  Bordes y esquinas en diferentes orientaciones
         ╱  ╲  ─  │  ╮  ╭

Capa 2:  Texturas y patrones simples
         ▓▓▓  ░░░  ╬╬╬  ≡≡≡

Capa 3:  Partes de objetos
         [ojo] [nariz] [rueda] [costilla]

Capa 4:  Objetos completos y conceptos
         [cara] [auto] [pulmón normal/enfermo]
```

Esta jerarquía fue visualizada empíricamente por Zeiler & Fergus (2014) en su trabajo "Visualizing and Understanding Convolutional Networks".

---

## 5. Función de activación ReLU

### El problema de la linealidad

Si solo usáramos convoluciones (operaciones lineales), apilar muchas capas sería **matemáticamente equivalente a una sola capa**:

```
Capa 1: y₁ = W₁ · x
Capa 2: y₂ = W₂ · y₁ = W₂ · W₁ · x = (W₂W₁) · x = W_efectivo · x
```

Dos capas lineales = una capa lineal con pesos diferentes. La profundidad no aportaría nada.

Necesitamos **no-linealidades** para que la red pueda aprender funciones complejas.

### ReLU: Rectified Linear Unit

```
ReLU(x) = max(0, x) = { x,  si x > 0
                       { 0,  si x ≤ 0
```

**Gráficamente:**

```
f(x)
  |         /
  |        /
  |       /
──┼──────/──────── x
  |
  (todo lo negativo se hace 0)
```

### ¿Por qué ReLU y no otras funciones?

Antes de ReLU se usaba **sigmoide** `σ(x) = 1/(1+e^{-x})` y **tanh**.

Problemas de sigmoide/tanh:
- **Vanishing gradient**: La derivada de sigmoide es máximo 0.25. En redes profundas, multiplicar gradientes pequeños muchas veces → gradiente ≈ 0 → los pesos no aprenden
- **Saturación**: Para valores grandes o pequeños de x, la derivada es ≈ 0 (la función se "aplana")
- **Computacionalmente costoso**: Involucra exponenciales

Ventajas de ReLU:
- **Sin saturación** para x > 0: el gradiente es siempre 1 para valores positivos
- **Computacionalmente trivial**: solo comparar con 0
- **Sparsity**: neuronas con activación negativa se "apagan" (output = 0), introduciendo dispersión en las representaciones
- **Convergencia más rápida**: Típicamente 6× más rápido que sigmoid en la práctica (Krizhevsky et al.)

### Variantes de ReLU

```
Leaky ReLU:  f(x) = max(0.01x, x)    ← no muere completamente
ELU:         f(x) = x si x>0, α(e^x - 1) si x≤0
GELU:        f(x) = x · Φ(x)         ← usado en Transformers (suave y continua)
Swish:       f(x) = x · σ(x)         ← descubierta por AutoML de Google
```

El problema de ReLU estándar es la "muerte de neuronas" (dying ReLU): si el gradiente lleva los pesos a un estado donde la neurona siempre tiene activación negativa, el gradiente es 0 y la neurona nunca se recupera. Leaky ReLU previene esto con un gradiente pequeño para valores negativos.

> **Idea pedagógica**: Comparar ReLU con un interruptor de luz con dimmer. Cuando la señal es positiva, la deja pasar (con su intensidad). Cuando es negativa, la apaga completamente. Esto es mucho más simple y efectivo que funciones suaves como sigmoid.

---

## 6. Capa de Pooling

### Propósito del Pooling

Después de una capa convolucional, tenemos mapas de características con la misma resolución que la entrada. El pooling realiza **submuestreo espacial**: reduce el tamaño de los mapas de características de forma controlada.

**¿Por qué reducir?**

1. **Reducción computacional**: Menos valores → menos cómputo en capas siguientes
2. **Invariancia a pequeñas traslaciones**: Al tomar el máximo/promedio de una región, el modelo tolera pequeños desplazamientos del patrón
3. **Ampliar campo receptivo**: Cada neurona en capas profundas "ve" una región más grande de la imagen original

### MaxPooling

```
Toma el valor MÁXIMO de cada ventana

Entrada (4×4):          MaxPool 2×2 → Salida (2×2):
┌───┬───┬───┬───┐       ┌───────┬───────┐
│ 7 │ 3 │ 5 │ 2 │       │       │       │
├───┼───┼───┼───┤       │   8   │   6   │
│ 8 │ 7 │ 1 │ 6 │       │       │       │
├───┼───┼───┼───┤       ├───────┼───────┤
│ 4 │ 9 │ 3 │ 9 │       │       │       │
├───┼───┼───┼───┤       │   9   │   9   │
│ 0 │ 8 │ 4 │ 5 │       │       │       │
└───┴───┴───┴───┘       └───────┴───────┘
                        (cuadrante TL: max(7,3,8,7)=8)
```

**¿Por qué el máximo?** Un mapa de características representa "dónde se detectó el patrón". El máximo preserva la activación más fuerte — dónde el patrón estaba más claramente presente — descartando activaciones débiles irrelevantes.

### GlobalAveragePooling2D

En lugar de una ventana deslizante, toma el **promedio de todo el mapa de características**:

```
Feature map (17×17):    Global Average:
┌──────────────────┐
│  0.1  0.8  0.2   │
│  0.9  0.7  0.3   │    →    promedio = 0.43
│  0.0  0.5  0.6   │
└──────────────────┘
```

Si tenemos 128 mapas de características de 17×17, GlobalAvgPool produce un vector de 128 valores. Esto evita la explosión de parámetros del Flatten.

### Flatten vs GlobalAvgPool — La diferencia crítica

```
Después de 3 bloques Conv+Pool, los feature maps son (17, 17, 128):

Flatten:              (17 × 17 × 128) = 36,992 valores
  → Dense(128): 36,992 × 128 = 4,735,104 parámetros  ← ENORME

GlobalAvgPool:        128 valores (promedio de cada mapa)
  → Dense(64): 128 × 64 = 8,192 parámetros  ← MANEJABLE
```

Con 120 imágenes de entrenamiento y 4.7M de parámetros: **pura memorización, no generalización**.

> **Intuición**: GlobalAvgPool pregunta "¿este tipo de patrón está presente en algún lugar de la imagen?" sin importar dónde exactamente. Es suficiente para clasificación y evita el sobreajuste con datasets pequeños.

---

## 7. Capas densas y clasificación final

### El clasificador final

Después de extraer características jerárquicas con los bloques Conv+Pool, necesitamos **combinarlas para tomar una decisión**. Esto lo hacen las capas densas (fully connected).

```
Representación de la imagen después de GlobalAvgPool:
[0.82, 0.03, 0.91, 0.15, 0.67, ..., 0.44]  ← vector de 128 valores

Cada valor responde a: "¿cuánto se detectó el patrón i en la imagen?"
La capa Dense combina estos valores con pesos aprendidos para predecir la clase.
```

### Función de activación final

**Para clasificación binaria (2 clases):**
```
Sigmoid: σ(x) = 1 / (1 + e^{-x})

Salida ∈ (0, 1), interpretada como P(clase = 1)

P(PNEUMONIA) = 0.85 → PNEUMONIA
P(PNEUMONIA) = 0.23 → NORMAL
```

**Para clasificación multiclase (k clases):**
```
Softmax: P(clase_i) = e^{x_i} / Σ_j e^{x_j}

Ejemplo con 3 clases:
Logits: [2.1, 0.3, -0.8]
Softmax: [0.78, 0.17, 0.05]  ← suman 1.0
```

### Dropout: regularización durante el entrenamiento

Dropout "apaga" aleatoriamente una fracción `p` de las neuronas durante cada paso de entrenamiento:

```
Sin Dropout:
[0.82, 0.03, 0.91, 0.15, 0.67]

Con Dropout(0.5) durante entrenamiento (la mitad se apaga aleatoriamente):
[0.82,   0 , 0.91,   0 , 0.67]

Durante inferencia: Dropout NO se aplica, todos los pesos se escalan por (1-p)
```

**¿Por qué funciona?** Fuerza a la red a no depender de ninguna neurona específica. Cada subconjunto de neuronas debe ser capaz de clasificar correctamente por sí mismo → robustez y generalización.

---

## 8. Arquitectura completa

### Anatomía de una CNN típica

```
┌─────────────────────────────────────────────────────────────┐
│                    ETAPA DE EXTRACCIÓN                       │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐          │
│  │ Bloque 1   │   │ Bloque 2   │   │ Bloque 3   │          │
│  │            │   │            │   │            │          │
│  │ Conv2D(32) │ → │ Conv2D(64) │ → │ Conv2D(128)│          │
│  │ ReLU       │   │ ReLU       │   │ ReLU       │          │
│  │ MaxPool2D  │   │ MaxPool2D  │   │ MaxPool2D  │          │
│  └────────────┘   └────────────┘   └────────────┘          │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    ETAPA DE CLASIFICACIÓN                    │
│  GlobalAvgPool2D → Dropout(0.3) → Dense(64) → Dense(1,σ)   │
└─────────────────────────────────────────────────────────────┘
```

### Dimensiones a través de la red (entrada 224×224×3)

| Capa | Operación | Forma de salida | Parámetros |
|------|-----------|-----------------|------------|
| Input | — | (224, 224, 3) | 0 |
| Conv2D(32, 3×3) | Convolución | (222, 222, 32) | 896 |
| MaxPool2D(2×2) | Submuestreo | (111, 111, 32) | 0 |
| Conv2D(64, 3×3) | Convolución | (109, 109, 64) | 18,496 |
| MaxPool2D(2×2) | Submuestreo | (54, 54, 64) | 0 |
| Conv2D(128, 3×3) | Convolución | (52, 52, 128) | 73,856 |
| MaxPool2D(2×2) | Submuestreo | (26, 26, 128) | 0 |
| GlobalAvgPool2D | Promedio global | (128,) | 0 |
| Dropout(0.3) | Regularización | (128,) | 0 |
| Dense(64) + ReLU | Clasificador | (64,) | 8,256 |
| Dense(1) + Sigmoid | Predicción | (1,) | 65 |
| **Total** | | | **101,569** |

### Campo receptivo (Receptive Field)

El **campo receptivo** es la región de la imagen original que influye sobre una neurona en una capa profunda.

```
Neurona en Conv1: ve una región 3×3 de la imagen
Neurona en Conv2: ve una región 5×5 de la imagen  (3×3 + stride)
Neurona en Conv3: ve una región 7×7 de la imagen

Con MaxPool (stride 2) entre cada capa, el campo receptivo efectivo crece mucho más rápido.
```

Las neuronas en capas profundas tienen un campo receptivo que abarca gran parte de la imagen, permitiéndoles detectar estructuras globales.

---

## 9. Entrenamiento: Backpropagation en CNNs

### Función de pérdida

Para clasificación binaria (NORMAL vs PNEUMONIA) usamos **Binary Cross-Entropy**:

```
L = -[y · log(ŷ) + (1-y) · log(1-ŷ)]

Donde:
  y  = etiqueta real (0=NORMAL, 1=PNEUMONIA)
  ŷ  = probabilidad predicha por el modelo

Si y=1 (PNEUMONIA) y ŷ=0.9: L = -log(0.9) = 0.105  ← pérdida baja (bien)
Si y=1 (PNEUMONIA) y ŷ=0.1: L = -log(0.1) = 2.303  ← pérdida alta (mal)
```

### Backpropagation

El entrenamiento ajusta los pesos para minimizar la pérdida media sobre todo el dataset:

```
Algoritmo:
1. Forward pass: imagen → predicción (usar pesos actuales)
2. Calcular pérdida: L(predicción, etiqueta_real)
3. Backward pass: calcular ∂L/∂W para cada peso W mediante la regla de la cadena
4. Actualizar pesos: W ← W - α · ∂L/∂W  (gradient descent)
5. Repetir para todos los batches (mini-batch gradient descent)
```

### Optimizador Adam

Adam (Adaptive Moment Estimation) adapta la tasa de aprendizaje por parámetro:

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t     ← momento de primer orden (media)
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²    ← momento de segundo orden (varianza)
W_t = W_{t-1} - α · m̂_t / (√v̂_t + ε)

Parámetros típicos: β₁=0.9, β₂=0.999, ε=1e-8, α=0.001
```

Adam combina las ventajas de AdaGrad (adapta el lr por parámetro) y RMSProp (tasa de aprendizaje decreciente). Es el optimizador más usado en la práctica.

### Curvas de aprendizaje y diagnóstico

Las curvas de entrenamiento (accuracy/loss en train vs validación) revelan el estado del aprendizaje:

```
Escenario 1 — Underfitting:
  Train acc: 60%,  Val acc: 58%
  → Modelo muy simple, necesita más capacidad o más épocas

Escenario 2 — Overfitting:
  Train acc: 99%,  Val acc: 65%
  → Modelo memoriza train, no generaliza. Necesita regularización.

Escenario 3 — Buen ajuste:
  Train acc: 92%,  Val acc: 89%
  → Gap pequeño, ambas curvas convergiendo → modelo aprendió bien

Escenario 4 — Colapso (lo que nos pasó con CNN desde cero):
  Train acc: 55%,  Val acc: 50%
  → Barely better than chance. El modelo no está aprendiendo NADA.
```

---

## 10. Transfer Learning

### El problema de fondo: no tenemos suficientes datos

Con solo **120 imágenes de entrenamiento**, una CNN desde cero:
- No puede aprender filtros útiles: hay más parámetros que ejemplos
- Colapsa a predecir siempre la misma clase
- Recall de PNEUMONIA: 0.15 (detecta 3 de 20 casos reales) — clínicamente peligroso

### ¿Qué es Transfer Learning?

Transfer Learning aprovecha el conocimiento que una red aprendió en una tarea/dataset grande para resolver una tarea diferente con menos datos.

```
Entrenamiento original:          Nueva tarea:
ImageNet (1.2M imágenes,    →   Rayos X COVID-19
1000 clases, semanas de         (120 imágenes,
entrenamiento en GPUs)          2 clases, minutos)

La red ya sabe detectar:        Solo necesita aprender:
  - Bordes y texturas            - ¿Esto parece pulmón normal?
  - Formas básicas               - ¿Hay consolidaciones?
  - Partes de objetos            - ¿Hay opacidades anómalas?
```

### ¿Por qué funciona?

Los primeros filtros de cualquier red entrenada en imágenes naturales aprenden básicamente los **mismos detectores de bajo nivel**: bordes, frecuencias, orientaciones. Estos son universales y útiles para cualquier problema de visión.

```
Visualización de filtros aprendidos en ImageNet (Zeiler & Fergus, 2014):
Capa 1: Detectores de bordes en 45°, 90°, 135°, 0° — colores básicos
Capa 2: Texturas, cuadrículas, manchas
Capa 3: Patrones más complejos, bordes de objetos
Capa 4: Partes de objetos (caras, ruedas, texto)
Capa 5: Objetos completos
```

Estos filtros de bajo y medio nivel son igualmente útiles para detectar bordes pulmonares, texturas de tejido sano/enfermo, etc.

### MobileNetV2: la arquitectura base

MobileNetV2 (Google, 2018) fue diseñada para eficiencia en dispositivos móviles:

**Inverted Residuals**: en lugar de reducir canales y luego expandir (como ResNet), expande primero y luego reduce. Esto preserva mejor la información en redes pequeñas.

```
Bloque de MobileNetV2:
Input (bajo canal) → Expansión 6× → DW Conv 3×3 → Proyección → Output (bajo canal)
                                                         ↑
                                               Skip connection si stride=1
```

**Depthwise Separable Convolution**: descompone la convolución estándar en dos pasos:
```
Convención estándar:  k × k × C_in × C_out parámetros
Depthwise separable:  k × k × C_in  +  C_in × C_out parámetros
                      ≈ 8-9× menos parámetros con similar precisión
```

### Estrategia de dos fases

#### Fase 1: Entrenar solo la cabeza

```
Base MobileNetV2:   [CONGELADA] — pesos de ImageNet fijos, no se actualizan
GlobalAvgPool       ←←←←←←←←←← solo estas capas
Dropout(0.3)        ←←←←←←←←←← se entrenan
Dense(1, Sigmoid)   ←←←←←←←←←← (1,281 parámetros entrenables)

Learning rate: 1e-3  (grande porque entrenar desde cero la cabeza)
Épocas: 15
```

La base ya tiene buenos features. La cabeza aprende a combinarlos para nuestro problema.

#### Fase 2: Fine-tuning de capas profundas

```
Base MobileNetV2 capas [0...-30]:   [CONGELADAS]
Base MobileNetV2 capas [-30:]:      [DESCONGELADAS] ← se ajustan levemente
Cabeza clasificadora:               [ENTRENANDO]

Learning rate: 1e-5  (muy pequeño para no destruir los pesos de ImageNet)
Épocas: 10
```

Las últimas capas de la base aprenden features específicos para rayos X, mientras las primeras capas (detectores de bordes básicos) se mantienen fijos.

### Resultado: la importancia del Transfer Learning con pocos datos

| Métrica | CNN desde cero | Transfer Learning |
|---------|---------------|-------------------|
| Parámetros entrenables (Fase 1) | 101,569 | **1,281** |
| Train accuracy (final) | 63% | 95% |
| Val accuracy (final) | 50% | 96% |
| Test accuracy | 57% | **97%** |
| Recall PNEUMONIA | **0.15** ❌ | **1.00** ✅ |
| Falsos negativos | 17/20 | 0/20 |

> **La lección más importante del ejemplo**: Con 120 imágenes, Transfer Learning no es una optimización — es la **única estrategia viable**. Una CNN desde cero literalmente no aprende nada.

---

## 11. Caso práctico: COVID-19 X-Ray

### El dataset

- **Fuente**: Khoong W.H. (2020). COVID-19 X-Ray Dataset — Kaggle
- **Clases**: NORMAL (pulmones sanos) vs PNEUMONIA (COVID-19)
- **Train**: 74 NORMAL + 74 PNEUMONIA = **148 imágenes**
- **Test**: 20 NORMAL + 20 PNEUMONIA = **40 imágenes**
- **Tamaño**: 84.3 MB

### Características de las imágenes

**NORMAL**: Campo pulmonar claro, sin opacidades, costillas bien definidas, diafragma visible.

**PNEUMONIA (COVID-19)**: Opacidades bilaterales, patrón en vidrio esmerilado (ground glass opacity), consolidaciones en regiones basales y periféricas.

### Pipeline completo

```
1. Descarga con kagglehub
        ↓
2. Preprocesamiento:
   - Redimensionar a 224×224
   - Aplicar preprocess_input de MobileNetV2 (normalizar a [-1, 1])
   - Data augmentation en entrenamiento:
     rotation_range=15, horizontal_flip=True, zoom_range=0.1
        ↓
3. Split: 80% train / 20% val (del conjunto de entrenamiento)
   → 120 imágenes train, 28 imágenes validación
        ↓
4. Transfer Learning Fase 1 (15 épocas, lr=1e-3)
        ↓
5. Fine-tuning Fase 2 (10 épocas, lr=1e-5)
        ↓
6. Evaluación en test (40 imágenes nunca vistas)
        ↓
7. Predicción con umbral=0.4 (conservador para minimizar falsos negativos)
```

### Decisión del umbral

El umbral por defecto es 0.5 (predice PNEUMONIA si la probabilidad > 50%). En diagnóstico médico:

```
Falso Positivo (FP): Decirle a alguien sano que tiene neumonía
  → Consecuencia: más exámenes, estrés, costo adicional

Falso Negativo (FN): Decirle a alguien con neumonía que está sano
  → Consecuencia: no recibe tratamiento, puede agravarse, riesgo de muerte

FN es MUCHO más grave que FP en este contexto.
```

Por eso usamos **umbral = 0.4**: predice PNEUMONIA si la probabilidad > 40%. Más conservadores sobre "declarar sano" a alguien → menos falsos negativos.

### Análisis de resultados

Con Transfer Learning:
```
              precision    recall  f1-score   support

     NORMAL       1.00      0.95      0.97        20
  PNEUMONIA       0.95      1.00      0.97        20

   accuracy                           0.97        40

Interpretación:
  - Recall PNEUMONIA = 1.00: los 20 casos de neumonía fueron detectados
  - Precisión NORMAL = 1.00: cuando predice NORMAL, siempre es correcto
  - El único error: 1 caso NORMAL predicho como PNEUMONIA (FP)
```

**Caveat estadístico importante**: 40 imágenes de test es un tamaño muy pequeño. El intervalo de confianza del 95% para una accuracy de 97% en 40 muestras es aproximadamente **[86%, 100%]**. Los resultados son prometedores pero no concluyentes para un uso clínico real.

---

## 12. Casos de uso reales

### Medicina y Salud

**Google Health (2020)**: CNN para detección de cáncer de mama en mamografías.
- Resultado: 94.5% de sensibilidad vs 88.0% del radiólogo promedio
- Reducción de 9.4% en falsos negativos vs panel de radiólogos

**Moorfields Eye Hospital / DeepMind**: CNN para diagnóstico de enfermedades oculares desde OCT (tomografía de coherencia óptica).
- 50+ enfermedades en un solo modelo
- Rendimiento equivalente a especialistas top del mundo

**COVID-Net (Wang & Wong, 2020)**: CNN específicamente diseñada para COVID-19.
- Entrenada en 13,645 radiografías
- 91% de sensibilidad para COVID-19

### Automotriz

**Tesla Autopilot**: Red de CNNs que procesa input de 8 cámaras en tiempo real.
- Detecta: vehículos, peatones, señales, carriles, obstáculos
- Latencia < 10ms para decisiones críticas
- Procesamiento: 144 TOPS (tera-operaciones por segundo) en el chip HW4

**Waymo**: Fusiona CNN con LIDAR y radar.
- 20 millones de millas en vehículos autónomos acumuladas

### Seguridad

**Apple Face ID**: CNN que mapea 30,000 puntos infrarrojos del rostro.
- 1 en 1,000,000 de probabilidad de desbloqueo incorrecto (vs 1 en 50,000 de Touch ID)
- Funciona en oscuridad, con gafas, sombrero, barba, etc.
- Reconoce cambios faciales graduales (envejecimiento, cirugías)

### Logística

**Amazon Robotics**: Sistemas de visión para almacenes automatizados.
- Identificación y agarre de millones de productos distintos
- Clasificación de paquetes a >1,000 unidades/hora

---

## 13. Ventajas y limitaciones

### Ventajas

| Ventaja | Explicación |
|---------|-------------|
| **Extracción automática de features** | No requiere ingeniería manual de características — el modelo aprende qué buscar |
| **Invariancia espacial** | Los filtros convolucionales detectan el mismo patrón independientemente de su posición |
| **Compartición de parámetros** | Eficiencia paramétrica vs redes fully connected |
| **Escalabilidad** | Se beneficia de más datos y más cómputo de forma predecible |
| **Transfer Learning** | Conocimiento transferible entre dominios → funciona con pocos datos |
| **Estado del arte** | Años de investigación la convierten en la arquitectura más madura para visión |

### Limitaciones

| Limitación | Explicación |
|------------|-------------|
| **Requiere datos** | Con < 1000 imágenes, el Transfer Learning es necesario pero no siempre suficiente |
| **Caja negra** | Difícil interpretar qué aprendió exactamente. Importante en medicina y derecho |
| **Sesgo del dataset** | Aprende sesgos presentes en los datos. Una CNN entrenada en hospitals de EE.UU. puede fallar en hospitales de otros países |
| **Costosa de entrenar** | Redes grandes (ResNet, EfficientNet) requieren semanas en clusters de GPUs |
| **Sensible a distribución** | Cambios en la distribución de datos (hospital diferente, equipo diferente) pueden degradar drásticamente el rendimiento |
| **No entiende contexto** | Puede clasificar correctamente por razones equivocadas (artefactos, marcas de agua, no el contenido clínico) |

### El problema de la interpretabilidad: Grad-CAM

**Grad-CAM** (Gradient-weighted Class Activation Mapping) genera **mapas de calor** que muestran qué regiones de la imagen influenciaron la decisión:

```
Predicción: PNEUMONIA (95%)
Mapa de calor: áreas más rojas = más influyentes en la decisión

Si el mapa de calor apunta a los pulmones → bien, el modelo mira lo correcto
Si el mapa de calor apunta a la marca del hospital → problema, el modelo aprendió un atajo
```

Esto es crítico para aplicaciones médicas donde necesitamos saber **por qué** el modelo tomó una decisión.

---

## 14. El futuro: más allá de las CNNs

### Vision Transformers (ViTs)

En 2020, Dosovitskiy et al. (Google Brain) demostró que los **Transformers** — la arquitectura que revolucionó el procesamiento de lenguaje (GPT, BERT) — también funcionan directamente sobre imágenes.

**Idea central**: Dividir la imagen en parches (patches) de 16×16 píxeles y tratarlos como tokens de una secuencia:

```
Imagen 224×224 → 196 parches de 16×16 → secuencia de 196 "tokens visuales"
                 → procesada por Transformer con atención global
```

**Diferencia clave con CNNs**:
- CNN: sesgo inductivo local (cada neurona solo ve su vecindad)
- ViT: atención global desde la primera capa (puede relacionar parches distantes)

Con suficientes datos (JFT-300M, 300 millones de imágenes), los ViTs superan a las CNNs.
Con pocos datos, las CNNs siguen siendo superiores por su sesgo inductivo local.

### Modelos Híbridos

La tendencia actual es combinar las fortalezas de ambas arquitecturas:

- **ConvNeXt (Meta, 2022)**: CNN "modernizada" con ideas de los Transformers (layer normalization, GELU, parcheo de 4×4, etc.). Compite con Swin Transformers.
- **Swin Transformer**: Transformer con ventanas jerárquicas que computa atención local, luego global. Balance eficiencia/rendimiento.
- **MobileViT**: Híbrido ultraligero de MobileNet + Transformer para dispositivos móviles.

### Edge AI: Inteligencia en el dispositivo

La tendencia es llevar los modelos directamente al hardware con recursos limitados:

**Técnicas de compresión**:
```
Cuantización: float32 (32 bits) → int8 (8 bits)
  → Modelo 4× más pequeño, 2-4× más rápido, ~1% pérdida de precisión

Pruning: Eliminar conexiones con pesos pequeños
  → Hasta 90% de sparsity con poca pérdida de precisión

Knowledge Distillation: Un modelo pequeño (estudiante) aprende a imitar
  uno grande (profesor) → modelo pequeño con comportamiento complejo
```

**Hardware especializado**:
- **Apple Neural Engine**: 35 TOPS, integrado en todos los chips M y A-series
- **Google Edge TPU**: Diseñado para inferencia de modelos cuantizados en edge
- **NVIDIA Jetson**: GPU embedded para robots, drones, sistemas industriales

---

## 15. Ideas pedagógicas para la exposición

### Estructura sugerida y tiempo estimado (60-75 min)

| Sección | Tiempo | Técnica |
|---------|--------|---------|
| Introducción + motivación | 5 min | Imagen de rayos X — pregunta al público |
| ¿Por qué no MLP? | 8 min | Cálculo en vivo de parámetros |
| Historia | 5 min | Timeline visual |
| Convolución | 12 min | Demo interactiva en la app |
| ReLU, Pooling | 6 min | Visualizaciones |
| Arquitectura completa | 5 min | Diagrama de flujo |
| Transfer Learning | 8 min | Comparación con/sin TL |
| Demo en vivo | 8 min | App de Streamlit |
| Casos de uso | 5 min | Tabla de industrias |
| Futuro | 3 min | ViTs, Edge AI |
| Conclusiones + preguntas | 5 min | |

### Analogías que funcionan bien

**Para la convolución**:
> "Imaginen que tienen una linterna pequeña que ilumina solo una región de 3×3 píxeles de la imagen. La mueven por toda la imagen, y en cada posición miden si hay un borde horizontal. El resultado es un mapa que dice 'aquí sí había un borde horizontal, aquí no'. Eso es un filtro convolucional."

**Para la jerarquía de features**:
> "Es como aprender a leer. Primero reconoces líneas y curvas básicas. Luego combinas esas líneas para reconocer letras. Luego combinas letras para reconocer palabras. Luego palabras para reconocer frases. Cada nivel construye sobre el anterior. Una CNN hace exactamente esto con las imágenes."

**Para Transfer Learning**:
> "Imaginen que quieren aprender a clasificar rayos X pero solo tienen 120 imágenes. Ahora imaginen que pueden contratar a un radiólogo que pasó años viendo millones de fotos — no de rayos X, sino de gatos, perros, autos, etc. Ya sabe perfectamente detectar bordes, texturas, formas. Solo necesita aprender qué combinación de esos patrones indica neumonía. Eso es Transfer Learning."

**Para ReLU**:
> "ReLU es como un filtro de Twitter que elimina todos los tweets negativos. Solo deja pasar los positivos tal como son, y todo lo negativo se convierte en silencio (0). Simple, pero extremadamente efectivo."

**Para Dropout**:
> "Imaginen que estudian para un examen pero cada día, aleatoriamente, deben estudiar sin algunos de sus libros. Al final del curso, son capaces de responder cualquier pregunta aunque no tengan todos los libros disponibles. Sus compañeros que siempre estudiaron con todos los libros fallan si les quitas uno. Eso es regularización mediante Dropout."

**Para el overfitting con datasets pequeños**:
> "120 imágenes de entrenamiento con 4.7 millones de parámetros es como intentar aprender a escribir una novela copiando exactamente 3 páginas. El modelo simplemente memoriza las 120 imágenes de entrenamiento y no sabe qué hacer con imágenes nuevas."

### Demostraciones interactivas durante la exposición

1. **Demo de convolución** (sección ⚙️ de la app):
   - Mostrar kernel de "detección de bordes horizontales" sobre la imagen
   - Cambiar a "detección de bordes verticales" — mostrar cómo cambia el mapa
   - Cambiar a "blur" — mostrar efecto de suavizado
   - Pregunta al público: "¿qué creen que hace este kernel?" antes de revelar la respuesta

2. **Demo de predicciones** (sección 🧪 de la app):
   - Cargar imágenes del conjunto de test
   - Mostrar predicciones con la barra de probabilidad
   - Cambiar el umbral de 0.5 a 0.4 y explicar el trade-off FP/FN
   - Buscar intencionalmente una imagen mal clasificada y discutir por qué

3. **Comparación visual CNN scratch vs Transfer Learning**:
   - Mostrar las curvas de entrenamiento (sección demo, tab "Modelo")
   - "Con CNN desde cero, el accuracy nunca sube del 63%. Con Transfer Learning, llega al 95% en pocas épocas"

### Preguntas que el público podría hacer y cómo responderlas

**"¿Por qué ReLU y no sigmoid en capas intermedias?"**
> ReLU no satura para valores positivos, mientras que sigmoid se aplana para valores grandes/pequeños, haciendo que el gradiente sea ~0 (vanishing gradient). Además ReLU es computacionalmente trivial — solo comparar con 0.

**"¿Cómo sabe la red qué filtros aprender?"**
> No los diseñamos. Los filtros se inicializan aleatoriamente y se ajustan automáticamente mediante backpropagation para minimizar el error de predicción. La red descubre sola que necesita detectar bordes, texturas, etc.

**"¿Por qué 97% de accuracy no es suficiente para uso clínico?"**
> Porque fue medido en solo 40 imágenes de test — estadísticamente débil. Un error en 40 imágenes es el 2.5% del dataset de test, pero no sabemos si ese porcentaje se mantendría en 10,000 imágenes. Además, el dataset viene de un solo origen (misma fuente, mismo equipo, mismo protocolo) — en el mundo real hay variabilidad enorme entre hospitales.

**"¿Qué es exactamente lo que aprende la red? ¿Puede equivocarse por razones incorrectas?"**
> Sí. Un ejemplo famoso: una CNN entrenada para detectar melanoma aprendió a identificar reglas de calibración que los dermatólogos colocan junto a lunares malignos. En ausencia de regla → predecía benigno. Esto es por qué se usan técnicas como Grad-CAM para verificar que la red mira la región correcta.

**"¿Por qué MobileNetV2 y no ResNet o VGG?"**
> MobileNetV2 tiene ~2.2M parámetros vs 138M de VGG16. Para nuestro hardware (sin GPU), el tamaño importa. Además, fue preentrenada en ImageNet con excelente rendimiento y está incluida en Keras. Para este demo, la eficiencia prima sobre el máximo rendimiento.

### Errores comunes que evitar al exponer

1. **No decir "la red entiende las imágenes"** — la red encuentra correlaciones estadísticas entre píxeles y etiquetas. No "entiende" en sentido cognitivo.

2. **No presentar los resultados del demo como definitivos** — explicar los límites estadísticos del dataset pequeño.

3. **No memorizar las fórmulas** — es mejor explicar la intuición y tener la fórmula escrita. Nadie espera que recites la derivada de la cross-entropy de memoria.

4. **No saltar la parte "¿por qué no MLP?"** — es crucial para que el público entienda la motivación de toda la arquitectura CNN. Sin este contexto, las CNNs parecen una solución sin un problema.

---

## 16. Referencias

### Artículos fundamentales

1. **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998)**. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
   - El paper original de LeNet-5. Introduce la arquitectura convolucional moderna.

2. **Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012)**. ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, 25.
   - El paper de AlexNet que marcó el inicio del Deep Learning moderno.

3. **He, K., Zhang, X., Ren, S., & Sun, J. (2016)**. Deep residual learning for image recognition. *Proceedings of CVPR*, 770–778.
   - ResNet: conexiones residuales para redes de 100+ capas sin vanishing gradient.

4. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018)**. MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of CVPR*.
   - La arquitectura base de nuestro modelo de Transfer Learning.

5. **Wang, L., & Wong, A. (2020)**. COVID-Net: A tailored deep convolutional neural network design for detection of COVID-19 cases from chest X-ray images. *Scientific Reports*, 10, 19549.
   - Inspiración directa para nuestro caso práctico.

6. **Dosovitskiy, A. et al. (2020)**. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.
   - El paper de Vision Transformers que desafía la dominancia de las CNNs.

### Recursos educativos

7. **Chollet, F. (2021)**. *Deep Learning with Python* (2nd ed.). Manning Publications.
   - El libro más práctico para aprender CNNs con Keras/TensorFlow.

8. **Zeiler, M. D., & Fergus, R. (2014)**. Visualizing and Understanding Convolutional Networks. *ECCV 2014*.
   - Cómo visualizar qué aprenden las CNNs. Base de Grad-CAM y similares.

### Dataset

9. **Khoong, W.H. (2020)**. COVID-19 X-Ray Dataset (Train/Test Sets). Kaggle.  
   `khoongweihao/covid19-xray-dataset-train-test-sets`

---

## Apéndice: Glosario rápido

| Término | Definición simple |
|---------|-------------------|
| **Kernel / Filtro** | Pequeña matriz de pesos que se desliza sobre la imagen para detectar un patrón |
| **Feature Map** | El resultado de aplicar un filtro a la imagen — un "mapa" de dónde se detectó el patrón |
| **Stride** | Cuántos píxeles se mueve el filtro en cada paso |
| **Padding** | Añadir ceros alrededor de la imagen para controlar el tamaño de la salida |
| **Epoch** | Una pasada completa sobre todo el dataset de entrenamiento |
| **Batch** | Subconjunto de imágenes procesadas juntas antes de actualizar los pesos |
| **Overfitting** | El modelo memoriza los datos de entrenamiento y falla en datos nuevos |
| **Underfitting** | El modelo es demasiado simple para capturar los patrones relevantes |
| **Gradient descent** | Algoritmo de optimización que ajusta los pesos en la dirección que reduce la pérdida |
| **Backpropagation** | Cálculo eficiente de los gradientes mediante la regla de la cadena |
| **Transfer Learning** | Reusar pesos preentrenados en una tarea distinta |
| **Fine-tuning** | Ajuste fino de los pesos preentrenados para la nueva tarea |
| **Recall** | De todos los casos positivos reales, ¿cuántos detectó el modelo? |
| **Precisión** | De todos los que el modelo predijo positivos, ¿cuántos realmente lo eran? |
| **Falso negativo** | El modelo predice negativo cuando la realidad es positiva |
| **Umbral** | Valor de probabilidad a partir del cual se clasifica como positivo |

---

*Documento generado para la exposición de Tópicos de Machine Learning — Universidad de Antioquia.*  
*Para correr la demostración: `streamlit run app.py` en el directorio del repositorio.*
