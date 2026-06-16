# CNN para Clasificación de Rayos X — COVID-19

**Universidad de Antioquia · Facultad de Ciencias Exactas y Naturales**  
Analítica de Datos — Tópicos de Machine Learning  
Profesor: Duván Cataño

---

## Descripción

Implementación de una **Red Neuronal Convolucional (CNN)** para clasificar radiografías de tórax en dos categorías: `NORMAL` vs `PNEUMONIA` (COVID-19).

El modelo aprende a detectar patrones visuales (opacidades, consolidaciones pulmonares) directamente de los píxeles, sin ingeniería manual de características.

---

## Estructura del Repositorio

```
Ml-CNN-Analtica_Datos/
├── data/
│   ├── xray_dataset_covid19/      # Dataset (no incluido en git — ver Datos)
│   │   ├── train/
│   │   │   ├── NORMAL/            # 74 imágenes
│   │   │   └── PNEUMONIA/         # 74 imágenes
│   │   └── test/
│   │       ├── NORMAL/            # 20 imágenes
│   │       └── PNEUMONIA/         # 20 imágenes
│   └── fig*.png                   # Figuras generadas por el notebook
├── notebooks/
│   └── cnn_covid19_xray.ipynb     # Implementación completa
├── env/                           # Entorno virtual Python (no incluido en git)
├── .gitignore
└── README.md
```

---

## Resultados

| Conjunto | Imágenes | Clases |
|---|---|---|
| Train | 148 | NORMAL / PNEUMONIA |
| Test | 40 | NORMAL / PNEUMONIA |

**Arquitectura CNN:**
```
Entrada (150×150×3)
→ Conv2D(32) → ReLU → MaxPool
→ Conv2D(64) → ReLU → MaxPool
→ Conv2D(128) → ReLU → MaxPool
→ Flatten → Dropout(0.5) → Dense(128) → Dense(1, sigmoid)
```

---

## Instalación y Uso

**1. Clonar el repositorio**
```bash
git clone <url-del-repo>
cd Ml-CNN-Analtica_Datos
```

**2. Crear entorno virtual e instalar dependencias**
```bash
python -m venv env
source env/bin/activate        # Linux/Mac
# env\Scripts\activate          # Windows

pip install tensorflow numpy matplotlib scikit-learn pillow seaborn jupyter streamlit plotly
```

**3. Descargar el dataset**
```python
import kagglehub
path = kagglehub.dataset_download("khoongweihao/covid19-xray-dataset-train-test-sets")
# Copiar contenido a data/xray_dataset_covid19/
```

**4. Ejecutar el notebook**
```bash
jupyter notebook notebooks/cnn_covid19_xray.ipynb
```

**5. Lanzar la aplicación Streamlit**
```bash
streamlit run app.py
```

---

## Contenido del Notebook

| Sección | Descripción |
|---|---|
| Introducción | Problema clínico de diagnóstico COVID-19 |
| Exploración | Visualización de muestras y distribución de clases |
| Preprocesamiento | Normalización + Data Augmentation |
| Arquitectura | CNN con 3 bloques convolucionales |
| Entrenamiento | Adam + EarlyStopping |
| Evaluación | Accuracy, F1-score, matriz de confusión |
| Visualización | Predicciones con imagen real |

---

## Referencias

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.
2. Wang, L., & Wong, A. (2020). COVID-Net: A tailored deep convolutional neural network design for detection of COVID-19 cases from chest X-ray images. *Scientific Reports*, 10, 19549.
3. Khoong, W.H. (2020). [COVID-19 X-Ray Dataset](https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets). Kaggle.
4. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.



