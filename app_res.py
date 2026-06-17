import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# ─── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redes Neuronales Convolucionales",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "xray_dataset_covid19"
MODEL_PATH = ROOT / "data" / "cnn_covid19.keras"

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1a2744;
    border-radius: 12px;
    padding: 20px;
    border-left: 4px solid #00d4ff;
    margin: 8px 0;
}
.concept-card {
    background: #1a2744;
    border-radius: 12px;
    padding: 18px;
    margin: 8px 0;
    border: 1px solid #2d4070;
}
.formula-box {
    background: #0a1628;
    border: 1px solid #00d4ff;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    font-family: monospace;
    font-size: 1.15em;
    color: #00d4ff;
    margin: 12px 0;
}
.timeline-item {
    background: #1a2744;
    border-radius: 10px;
    padding: 16px;
    border-top: 3px solid #00d4ff;
    margin: 4px;
    text-align: center;
}
.section-header {
    background: linear-gradient(135deg, #0d1b2a, #1a2744);
    border-radius: 12px;
    padding: 24px;
    border-bottom: 3px solid #00d4ff;
    margin-bottom: 20px;
}
.pred-correct   { color: #4caf50; font-weight: bold; }
.pred-incorrect { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def card(title: str, body: str, border: str = "#00d4ff"):
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:{border};">
        <h4 style="color:{border};margin:0 0 8px 0;">{title}</h4>
        <p style="color:#e0e0e0;margin:0;line-height:1.6;">{body}</p>
    </div>""", unsafe_allow_html=True)


def concept_card(icon: str, title: str, body: str):
    st.markdown(f"""
    <div class="concept-card">
        <h4 style="color:#00d4ff;margin:0 0 8px 0;">{icon} {title}</h4>
        <p style="color:#cdd5e0;margin:0;line-height:1.6;">{body}</p>
    </div>""", unsafe_allow_html=True)


def setup_ax(ax, title=""):
    ax.set_facecolor("#1a2744")
    ax.tick_params(colors="#e0e0e0", labelsize=9)
    for s in ax.spines.values():
        s.set_color("#2d4070")
    ax.xaxis.label.set_color("#e0e0e0")
    ax.yaxis.label.set_color("#e0e0e0")
    if title:
        ax.set_title(title, color="#00d4ff", fontsize=11, fontweight="bold", pad=8)


def dark_fig(rows=1, cols=1, **kw):
    fig, axes = plt.subplots(rows, cols, **kw)
    fig.patch.set_facecolor("#0d1b2a")
    return fig, axes


# ─── MODEL (cached) ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo…")
def load_model():
    if not MODEL_PATH.exists():
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(str(MODEL_PATH))


# ══════════════════════════════════════════════════════════════════════════════
# SECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def sec_inicio():
    st.markdown("""
    <div class="section-header">
        <h1 style="color:#00d4ff;margin:0;">🧠 Redes Neuronales Convolucionales</h1>
        <h3 style="color:#e0e0e0;margin:6px 0 0 0;font-weight:normal;">
            Arquitectura, Historia y el Futuro de la Visión Artificial
        </h3>
        <p style="color:#8899aa;margin:8px 0 0 0;">
            Universidad de Antioquia · Analítica de Datos · Profesor: Duván Cataño
        </p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        card("🎯 Objetivo",
             "Comprender cómo las CNNs resuelven el reconocimiento visual de forma eficiente y jerárquica, aprendiendo características directamente de los píxeles.")
    with c2:
        card("📚 Contenido",
             "Fundamentos teóricos, arquitectura capa a capa, evolución histórica, casos de uso industriales y demostración con rayos X de COVID-19.")
    with c3:
        card("🧪 Demo práctica",
             "Clasificación de radiografías de tórax (NORMAL vs PNEUMONIA) usando Transfer Learning con MobileNetV2, logrando 97% de precisión.")

    st.divider()
    st.markdown("### ¿Qué es una CNN en una frase?")
    st.markdown("""
    <div class="formula-box">
        Una CNN aprende <em>qué buscar</em> en una imagen, dónde buscarlo,
        y cómo combinar lo que encuentra para tomar una decisión.
    </div>""", unsafe_allow_html=True)

    # Architecture overview figure
    fig, ax = dark_fig(1, 1, figsize=(12, 2.8))
    ax.axis("off")
    ax.set_facecolor("#0d1b2a")
    blks = [
        ("Imagen\n224×224×3", "#1565C0"),
        ("Conv\nReLU\n→ bordes", "#00838F"),
        ("MaxPool\n↓ espacio", "#00695C"),
        ("Conv\nReLU\n→ formas", "#2E7D32"),
        ("MaxPool\n↓ espacio", "#558B2F"),
        ("Global\nAvgPool", "#F57F17"),
        ("Dense\nSigmoid", "#D84315"),
        ("NORMAL\n/ PNEUMONIA", "#B71C1C"),
    ]
    xs = np.linspace(0.04, 0.96, len(blks))
    hs = [0.78, 0.65, 0.50, 0.60, 0.45, 0.38, 0.32, 0.28]
    for i, ((lbl, col), x, h) in enumerate(zip(blks, xs, hs)):
        r = patches.FancyBboxPatch(
            (x - 0.052, 0.5 - h / 2), 0.1, h,
            boxstyle="round,pad=0.01", facecolor=col,
            alpha=0.85, transform=ax.transAxes, zorder=2
        )
        ax.add_patch(r)
        ax.text(x, 0.5, lbl, transform=ax.transAxes,
                ha="center", va="center", color="white",
                fontsize=7.2, fontweight="bold", zorder=3)
        if i < len(blks) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.057, 0.5),
                        xytext=(x + 0.057, 0.5),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#00d4ff", lw=1.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def sec_por_que_no_mlp():
    st.markdown('<div class="section-header"><h2 style="color:#00d4ff;margin:0;">❓ ¿Por qué no usar Redes Tradicionales (MLP)?</h2></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💥 La Pesadilla Computacional", "🗺️ Pérdida de Estructura Espacial"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### El problema de escala")
            st.markdown("""
            Considera una imagen **estándar de 256×256 píxeles** a color (RGB):
            """)
            st.markdown("""
            <div class="metric-card" style="text-align:center;">
                <h1 style="color:#00d4ff;font-size:3.5em;margin:0;">196,608</h1>
                <p style="color:#e0e0e0;">Neuronas de entrada (256 × 256 × 3)</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("""
            <div class="metric-card" style="text-align:center;border-left-color:#f44336;">
                <h1 style="color:#f44336;font-size:3.5em;margin:0;">&gt; 196M</h1>
                <p style="color:#e0e0e0;">Pesos solo en la primera capa oculta (con 1,000 neuronas)</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("""
            <div class="concept-card">
                <p style="color:#cdd5e0;">Con una sola capa oculta pequeña de 1,000 neuronas, el modelo se vuelve
                computacionalmente <strong style="color:#f44336;">inmanejable</strong>, propenso al overfitting extremo
                y lento en converger.</p>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown("### Comparación de parámetros")
            fig, ax = dark_fig(1, 1, figsize=(6, 4))
            models = ["MLP\n(imagen 256×256)", "CNN básica\n(3 bloques)", "MobileNetV2\n(Transfer L.)"]
            params = [196_608_000, 101_569, 2_259_265]
            colors = ["#f44336", "#4caf50", "#00d4ff"]
            bars = ax.barh(models, params, color=colors, alpha=0.85)
            ax.set_xscale("log")
            for bar, p in zip(bars, params):
                ax.text(p * 1.5, bar.get_y() + bar.get_height() / 2,
                        f"{p:,}", va="center", color="white", fontsize=9)
            setup_ax(ax, "Número de parámetros (escala log)")
            ax.set_xlabel("Parámetros", color="#e0e0e0")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### El proceso de Flattening destruye información")
            concept_card("🔴", "Destrucción topológica",
                         "Al convertir una imagen 2D en un vector 1D, se pierde la relación de proximidad espacial entre píxeles vecinos.")
            concept_card("🔴", "Sin invariancia a traslación",
                         "Un objeto desplazado 2 píxeles se lee como un patrón completamente nuevo, requiriendo re-aprendizaje.")
            concept_card("🔴", "Incapacidad de detectar patrones locales",
                         "Imposible detectar bordes, texturas y contornos de forma eficiente porque se pierde la vecindad espacial.")
            concept_card("🟢", "Solución: Convolución",
                         "Los filtros convolucionales operan directamente sobre la estructura 2D, preservando y explotando la vecindad espacial.")

        with c2:
            st.markdown("### Visualización del Flattening")
            fig, axes = dark_fig(1, 2, figsize=(8, 4))
            np.random.seed(42)
            img = np.random.randint(50, 220, (5, 5, 3))
            axes[0].imshow(img)
            axes[0].set_title("Imagen 2D (5×5)", color="#00d4ff", fontsize=11)
            axes[0].set_facecolor("#1a2744")
            for i in range(5):
                for j in range(5):
                    axes[0].text(j, i, f"p{i*5+j}", ha="center", va="center",
                                 color="white", fontsize=7, fontweight="bold")
            flat = img.flatten()
            axes[1].bar(range(len(flat)), flat, color="#f44336", alpha=0.6, width=1.0)
            axes[1].set_title("Vector 1D — sin estructura espacial", color="#f44336", fontsize=10)
            axes[1].set_xlabel("Índice del pixel")
            axes[1].set_ylabel("Valor")
            for ax in axes:
                setup_ax(ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def sec_historia():
    st.markdown('<div class="section-header"><h2 style="color:#00d4ff;margin:0;">📅 Evolución e Hitos Fundacionales</h2></div>', unsafe_allow_html=True)

    hitos = [
        ("1959", "Hubel & Wiesel", "#1565C0",
         "Descubren que la corteza visual de los mamíferos procesa información de forma jerárquica: células simples responden a bordes y orientaciones locales, células complejas a patrones más abstractos."),
        ("1980", "Neocognitrón", "#00695C",
         "Kunihiko Fukushima propone el Neocognitrón: primer modelo computacional inspirado en la jerarquía visual de Hubel & Wiesel. Introduce la idea de capas de detección y submuestreo."),
        ("1998", "LeNet-5", "#2E7D32",
         "Yann LeCun presenta LeNet-5, primera CNN viable para producción. Reconoce dígitos escritos a mano en cheques bancarios para AT&T. Define la arquitectura Conv → Pool → Conv → Pool → Dense."),
        ("2012", "AlexNet", "#E65100",
         "Alex Krizhevsky gana ImageNet con 15.3% de error (vs 26.2% del segundo). Usa GPUs, activación ReLU y Dropout. Marca el inicio de la era moderna del Deep Learning."),
        ("2014", "VGGNet / GoogLeNet", "#BF360C",
         "VGG (Oxford) demuestra que la profundidad importa: 16-19 capas de filtros 3×3. GoogLeNet introduce los módulos Inception para eficiencia computacional."),
        ("2015+", "ResNet & más", "#B71C1C",
         "ResNet (Microsoft) introduce conexiones residuales: 152 capas sin vanishing gradient, superando a humanos en ImageNet. Abre camino a DenseNet, EfficientNet, y arquitecturas modernas."),
    ]

    # Timeline visual
    fig, ax = dark_fig(1, 1, figsize=(13, 2.5))
    ax.axis("off")
    ax.set_facecolor("#0d1b2a")
    years = [h[0] for h in hitos]
    colors = [h[2] for h in hitos]
    xs = np.linspace(0.06, 0.94, len(hitos))
    ax.axhline(0.5, color="#2d4070", lw=2)
    for i, (x, (yr, name, col, _)) in enumerate(zip(xs, hitos)):
        y = 0.75 if i % 2 == 0 else 0.25
        ax.annotate("", xy=(x, 0.5), xytext=(x, y + (0.05 if i % 2 == 0 else -0.05)),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", color=col, lw=2))
        circ = plt.Circle((x, 0.5), 0.025, color=col, transform=ax.transAxes, zorder=5)
        ax.add_patch(circ)
        ax.text(x, y + (0.1 if i % 2 == 0 else -0.1), yr,
                transform=ax.transAxes, ha="center", va="center",
                color=col, fontsize=10, fontweight="bold")
        ax.text(x, y + (0.0 if i % 2 == 0 else -0.22), name,
                transform=ax.transAxes, ha="center", va="center",
                color="#e0e0e0", fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()
    cols = st.columns(3)
    for i, (yr, name, col, desc) in enumerate(hitos):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="timeline-item" style="border-top-color:{col};">
                <h3 style="color:{col};margin:0;">{yr}</h3>
                <h4 style="color:#e0e0e0;margin:4px 0;">{name}</h4>
                <p style="color:#8899aa;font-size:0.85em;margin:0;">{desc}</p>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
def sec_convolucion():
    st.markdown('<div class="section-header"><h2 style="color:#00d4ff;margin:0;">⚙️ Capa Convolucional — El Núcleo de las CNNs</h2></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### ¿Qué es la convolución?")
        st.markdown("""
        Una capa convolucional aplica un **filtro (kernel)** deslizante sobre la imagen,
        calculando el producto punto entre el kernel y cada parche local de la imagen.

        El resultado es un **mapa de características** que indica dónde y cuánto
        cada patrón del kernel aparece en la imagen.
        """)
        st.markdown("""<div class="formula-box">
            S(i, j) = (I * W)(i, j) = Σ<sub>m</sub> Σ<sub>n</sub> I(i+m, j+n) · W(m, n)
        </div>""", unsafe_allow_html=True)
        concept_card("🔁", "Compartición de pesos",
                     "Un único filtro procesa <em>toda</em> la imagen. Esto reduce drásticamente el número de parámetros vs una capa densa.")
        concept_card("📐", "Invariancia a traslación",
                     "Si el filtro detecta un borde en la esquina superior, detectará el mismo borde en cualquier otra posición.")
        concept_card("🏗️", "Jerarquía de features",
                     "Capas tempranas: bordes y esquinas. Capas medias: texturas y formas. Capas profundas: partes de objetos y conceptos semánticos.")

    with c2:
        st.markdown("### Demo interactiva de convolución")
        kernels = {
            "Detección de bordes horizontal": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float),
            "Detección de bordes vertical": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float),
            "Suavizado (blur)": np.ones((3, 3)) / 9,
            "Realce (sharpen)": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
            "Detección de esquinas": np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float),
        }
        sel = st.selectbox("Elige un filtro kernel:", list(kernels.keys()))
        kernel = kernels[sel]

        np.random.seed(7)
        inp = np.array([
            [200, 180, 160, 140, 120, 100],
            [195, 175, 155, 135, 115,  95],
            [190,  50,  50,  50,  50,  90],
            [185,  50, 200, 200,  50,  85],
            [180,  50,  50,  50,  50,  80],
            [175, 155, 135, 115,  95,  75],
        ], dtype=float)

        out = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                out[i, j] = np.sum(inp[i:i+3, j:j+3] * kernel)

        fig, axes = dark_fig(1, 3, figsize=(10, 3.5))

        # Input
        axes[0].imshow(inp, cmap="Blues", vmin=0, vmax=255, aspect="auto")
        setup_ax(axes[0], "Entrada (6×6)")
        for i in range(6):
            for j in range(6):
                axes[0].text(j, i, str(int(inp[i, j])), ha="center", va="center",
                             color="white", fontsize=8)
        rect = patches.Rectangle((-0.5, -0.5), 3, 3, lw=2, edgecolor="#00d4ff", facecolor="none")
        axes[0].add_patch(rect)
        axes[0].text(1, 6.6, "← kernel aquí", color="#00d4ff", ha="center", fontsize=8)

        # Kernel
        vmax = max(abs(kernel.max()), abs(kernel.min()), 0.01)
        axes[1].imshow(kernel, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        setup_ax(axes[1], f"Kernel 3×3")
        for i in range(3):
            for j in range(3):
                v = kernel[i, j]
                txt = f"{v:.2f}" if abs(v) < 10 else str(int(v))
                axes[1].text(j, i, txt, ha="center", va="center",
                             color="white" if abs(v) > vmax * 0.4 else "black",
                             fontsize=11, fontweight="bold")

        # Output
        out_clip = np.clip(out, 0, 255)
        axes[2].imshow(out_clip, cmap="Greens", vmin=0, vmax=255, aspect="auto")
        setup_ax(axes[2], "Mapa de características (4×4)")
        for i in range(4):
            for j in range(4):
                axes[2].text(j, i, str(int(out[i, j])), ha="center", va="center",
                             color="white", fontsize=9)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("El kernel se desliza sobre la imagen, multiplicando elemento a elemento y sumando → un valor por posición.")


# ─────────────────────────────────────────────────────────────────────────────
def sec_activacion():
    st.markdown('<div class="section-header"><h2 style="color:#00d4ff;margin:0;">⚡ Función de Activación — ReLU</h2></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### ¿Por qué necesitamos no-linealidad?")
        st.markdown("""
        Sin funciones de activación, apilar capas lineales (convolución, multiplicación de matrices)
        equivale a **una sola capa lineal**, sin importar cuántas capas haya.

        La no-linealidad permite modelar **relaciones complejas** en los datos.
        """)
        concept_card("⚡", "ReLU — Rectified Linear Unit",
                     "<strong>f(x) = max(0, x)</strong> — Deja pasar valores positivos, elimina negativos. Simple, efectiva y eficiente.")
        concept_card("💡", "Ventajas de ReLU",
                     """
                     <ul>
                     <li>Evita el <em>vanishing gradient</em> de sigmoide/tanh</li>
                     <li>Computacionalmente trivial (solo comparar con 0)</li>
                     <li>Introduce <em>sparsity</em>: apaga neuronas irrelevantes</li>
                     <li>Convergencia más rápida en la práctica</li>
                     </ul>
                     """)
        concept_card("🔧", "Variantes modernas",
                     "Leaky ReLU: f(x) = max(0.01x, x) — no muere del todo. ELU, GELU (usado en Transformers). Pero ReLU sigue siendo el estándar en CNNs.")

        st.markdown("""<div class="formula-box">
            ReLU: f(x) = max(0, x)  →  { x  si x > 0<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ 0  si x ≤ 0
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("### Visualización de ReLU y comparación")
        x = np.linspace(-3, 3, 300)
        activaciones = {
            "ReLU":    (np.maximum(0, x), "#00d4ff"),
            "Sigmoid": (1 / (1 + np.exp(-x)), "#FF9800"),
            "Tanh":    (np.tanh(x), "#4CAF50"),
            "Leaky ReLU": (np.where(x > 0, x, 0.1 * x), "#9C27B0"),
        }
        fig, axes = dark_fig(1, 2, figsize=(10, 4))

        # ReLU solo
        axes[0].plot(x, np.maximum(0, x), color="#00d4ff", lw=3, label="ReLU")
        axes[0].axhline(0, color="#2d4070", lw=1)
        axes[0].axvline(0, color="#2d4070", lw=1)
        axes[0].fill_between(x, 0, np.maximum(0, x), alpha=0.15, color="#00d4ff")
        axes[0].annotate("x > 0 → pasa", xy=(2, 2), xytext=(0.8, 2.3),
                         color="#4caf50", fontsize=10, fontweight="bold")
        axes[0].annotate("x ≤ 0 → se apaga (0)", xy=(-2, 0), xytext=(-2.8, 0.5),
                         color="#f44336", fontsize=10, fontweight="bold")
        setup_ax(axes[0], "f(x) = max(0, x)")

        # Comparación
        for name, (vals, col) in activaciones.items():
            axes[1].plot(x, vals, label=name, color=col, lw=2)
        axes[1].axhline(0, color="#2d4070", lw=1)
        axes[1].axvline(0, color="#2d4070", lw=1)
        axes[1].legend(facecolor="#1a2744", labelcolor="#e0e0e0", fontsize=9)
        setup_ax(axes[1], "Comparación de activaciones")

        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("### Efecto de ReLU en un mapa de características")
        fig2, axes2 = dark_fig(1, 2, figsize=(8, 3))
        np.random.seed(0)
        fmap = np.random.randn(6, 6) * 100
        axes2[0].imshow(fmap, cmap="RdBu_r", vmin=-150, vmax=150, aspect="auto")
        setup_ax(axes2[0], "Antes de ReLU (valores negativos y positivos)")
        axes2[0].set_xticks([]); axes2[0].set_yticks([])

        axes2[1].imshow(np.maximum(0, fmap), cmap="Blues", vmin=0, vmax=150, aspect="auto")
        setup_ax(axes2[1], "Después de ReLU (solo activaciones positivas)")
        axes2[1].set_xticks([]); axes2[1].set_yticks([])
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
def sec_pooling():
    st.markdown('<div class="section-header"><h2 style="color:#00d4ff;margin:0;">🗜️ Capa de Agrupamiento (Pooling)</h2></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### ¿Por qué reducir el espacio?")
        concept_card("📦", "Eficiencia computacional",
                     "Reducir resolución espacial disminuye el costo de las capas siguientes. Un MaxPool 2×2 reduce 4× el número de valores.")
        concept_card("🔍", "Invariancia posicional",
                     "Al tomar el máximo en una ventana, el modelo tolera pequeñas traslaciones del objeto sin perder la detección.")
        concept_card("🌐", "Ampliar campo receptivo",
                     "Cada neurona en capas profundas 've' una región más grande de la imagen original, permitiendo detectar objetos más grandes.")

        st.markdown("### Tipos de Pooling")
        col_a, col_b = st.columns(2)
        with col_a:
            card("Max Pooling", "Toma el valor <strong>máximo</strong> de la ventana. Preserva los rasgos más dominantes. El más usado.", border="#00d4ff")
        with col_b:
            card("Average Pooling", "Toma el <strong>promedio</strong> de la ventana. Más suave, útil en capas finales (GlobalAvgPool).", border="#4caf50")

        st.markdown("""<div class="formula-box">
            MaxPool 2×2: O(i,j) = max{ I(2i,2j), I(2i+1,2j), I(2i,2j+1), I(2i+1,2j+1) }
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("### Visualización paso a paso")
        inp = np.array([[7, 3, 5, 2], [8, 7, 1, 6], [4, 9, 3, 9], [0, 8, 4, 5]])
        out_max = np.array([[8, 6], [9, 9]])
        out_avg = np.array([[inp[:2, :2].mean(), inp[:2, 2:].mean()],
                            [inp[2:, :2].mean(), inp[2:, 2:].mean()]])

        pool_type = st.radio("Tipo de pooling:", ["Max Pooling", "Average Pooling"], horizontal=True)
        out = out_max if pool_type == "Max Pooling" else out_avg

        fig, axes = dark_fig(1, 2, figsize=(8, 4))
        colors_input = np.zeros((4, 4, 3))
        quad_colors = [
            np.array([0.1, 0.4, 0.8]),
            np.array([0.8, 0.3, 0.1]),
            np.array([0.1, 0.6, 0.3]),
            np.array([0.6, 0.1, 0.6]),
        ]
        positions = [((0, 0), (0, 2)), ((0, 2), (0, 2)), ((2, 0), (2, 2)), ((2, 2), (2, 2))]
        for (r, c), col in zip([(0, 0), (0, 2), (2, 0), (2, 2)], quad_colors):
            colors_input[r:r+2, c:c+2] = col

        axes[0].imshow(colors_input, aspect="auto", alpha=0.6)
        for i in range(4):
            for j in range(4):
                axes[0].text(j, i, str(inp[i, j]), ha="center", va="center",
                             color="white", fontsize=16, fontweight="bold")
        for r, c in [(1.5, -0.5), (1.5, 1.5)]:
            axes[0].axhline(r, color="white", lw=2)
        for r, c in [(-0.5, 1.5), (1.5, 1.5)]:
            axes[0].axvline(c, color="white", lw=2)
        setup_ax(axes[0], "Entrada (4×4) — ventana 2×2")
        axes[0].set_xticks([]); axes[0].set_yticks([])

        colors_output = np.array([quad_colors[:2], quad_colors[2:]])
        axes[1].imshow(colors_output, aspect="auto", alpha=0.6)
        for i in range(2):
            for j in range(2):
                v = out[i, j]
                txt = str(int(v)) if v == int(v) else f"{v:.1f}"
                axes[1].text(j, i, txt, ha="center", va="center",
                             color="white", fontsize=20, fontweight="bold")
        label = "máximo" if pool_type == "Max Pooling" else "promedio"
        setup_ax(axes[1], f"Salida (2×2) — {label} por cuadrante")
        axes[1].set_xticks([]); axes[1].set_yticks([])

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(f"Cada color representa un cuadrante 2×2. {pool_type} toma el {label} de cada uno → reduce de 4×4 a 2×2.")


# ─────────────────────────────────────────────────────────────────────────────
def sec_capas_densas():
    st.markdown('<div class="section-header"><h2 style="color:#00d4ff;margin:0;">🔗 Capas Densas y Clasificación Final</h2></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Del mapa de características a la predicción")
        st.markdown("""
        Después de los bloques Conv+Pool, la red ha extraído **representaciones abstractas** de la imagen.
        Las capas densas usan estas representaciones para tomar la decisión final de clasificación.
        """)
        c_a, c_b, c_c = st.columns(3)
        with c_a:
            card("1️⃣ Global Avg Pool",
                 "Promedia cada mapa de características a <strong>un solo valor</strong>. Reemplaza el Flatten para reducir parámetros.",
                 border="#00d4ff")
        with c_b:
            card("2️⃣ Capa Densa",
                 "Conecta todas las entradas con todas las salidas con pesos entrenables. Combina features abstractos.",
                 border="#4caf50")
        with c_c:
            card("3️⃣ Activación Final",
                 "<strong>Sigmoid</strong> para clasificación binaria (0-1). <strong>Softmax</strong> para múltiples clases (suma=1).",
                 border="#FF9800")

        st.markdown("### Flatten vs GlobalAveragePooling2D")
        concept_card("❌", "Flatten (problemático con pocos datos)",
                     "Convierte (17×17×128) → 36,992 valores → Dense(128) = <strong>4.7M parámetros</strong>. Con 120 imágenes: memorización pura.")
        concept_card("✅", "GlobalAveragePooling2D (recomendado)",
                     "Promedia cada mapa de 17×17 a 1 valor → (128,) → Dense(64) = <strong>8,256 parámetros</strong>. Mucho más regularizado.")

        st.markdown("""<div class="formula-box">
            Sigmoid: σ(x) = 1 / (1 + e<sup>−x</sup>)<br>
            Softmax: σ(x)<sub>i</sub> = e<sup>x<sub>i</sub></sup> / Σ<sub>j</sub> e<sup>x<sub>j</sub></sup>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("### Visualización del flujo final")
        fig, ax = dark_fig(1, 1, figsize=(6, 8))
        ax.axis("off")
        ax.set_facecolor("#0d1b2a")

        stages = [
            ((17, 17, 128), "Feature Maps\n(17×17×128)", "#1565C0", 0.9),
            ((1, 128), "GlobalAvgPool\n→ (128,)", "#00695C", 0.72),
            ((1, 64), "Dense(64) + ReLU\n→ (64,)", "#2E7D32", 0.55),
            ((1, 32), "Dropout(0.3)\n→ (64,)", "#795548", 0.40),
            ((1, 1), "Dense(1) + Sigmoid\n→ probabilidad", "#D84315", 0.25),
            ((1, 1), "Predicción\nNORMAL / PNEUMONIA", "#B71C1C", 0.10),
        ]

        for (_, label, color, y) in stages:
            h = 0.1
            r = patches.FancyBboxPatch((0.1, y - h/2), 0.8, h,
                                        boxstyle="round,pad=0.01",
                                        facecolor=color, alpha=0.85,
                                        transform=ax.transAxes)
            ax.add_patch(r)
            ax.text(0.5, y, label, transform=ax.transAxes,
                    ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")

        for i in range(len(stages) - 1):
            y1 = stages[i][3] - 0.05
            y2 = stages[i+1][3] + 0.05
            ax.annotate("", xy=(0.5, y2), xytext=(0.5, y1),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#00d4ff", lw=2))

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def sec_flujo_completo():
    st.markdown('<div class="section-header"><h2 style="color:#00d4ff;margin:0;">🔄 Flujo Completo de una CNN</h2></div>', unsafe_allow_html=True)

    st.markdown("### Arquitectura CNN — Forward Pass completo")

    fig, ax = dark_fig(1, 1, figsize=(14, 5))
    ax.axis("off")
    ax.set_facecolor("#0d1b2a")

    layers = [
        ("Input\n224×224×3", "#1565C0", 1.0),
        ("Conv2D\n32 filtros\n3×3", "#0277BD", 0.85),
        ("ReLU", "#00695C", 0.55),
        ("MaxPool\n2×2", "#00695C", 0.70),
        ("Conv2D\n64 filtros\n3×3", "#1B5E20", 0.75),
        ("ReLU", "#388E3C", 0.50),
        ("MaxPool\n2×2", "#388E3C", 0.62),
        ("Conv2D\n128 filtros\n3×3", "#E65100", 0.68),
        ("ReLU", "#BF360C", 0.45),
        ("MaxPool\n2×2", "#BF360C", 0.58),
        ("GlobalAvg\nPool2D", "#880E4F", 0.45),
        ("Dropout\n0.3", "#4A148C", 0.38),
        ("Dense\n64+ReLU", "#311B92", 0.32),
        ("Dense\n1+Sigmoid", "#B71C1C", 0.28),
        ("Output\nP(pneum.)", "#C62828", 0.22),
    ]

    xs = np.linspace(0.02, 0.98, len(layers))

    for i, (lbl, col, h) in enumerate(layers):
        x = xs[i]
        r = patches.FancyBboxPatch(
            (x - 0.028, 0.5 - h/2), 0.055, h,
            boxstyle="round,pad=0.005",
            facecolor=col, alpha=0.85, transform=ax.transAxes, zorder=2
        )
        ax.add_patch(r)
        ax.text(x, 0.5, lbl, transform=ax.transAxes,
                ha="center", va="center", color="white",
                fontsize=6.5, fontweight="bold", zorder=3)
        if i < len(layers) - 1:
            ax.annotate("", xy=(xs[i+1] - 0.032, 0.5),
                        xytext=(x + 0.032, 0.5),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#00d4ff", lw=1.5))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()
    st.markdown("### Dimensiones a través de la red")
    dims = [
        ("Entrada", "224 × 224 × 3", "—", "150,528"),
        ("Conv2D(32) + MaxPool", "74 × 74 × 32", "896 params", "175,232"),
        ("Conv2D(64) + MaxPool", "36 × 36 × 64", "18,496 params", "82,944"),
        ("Conv2D(128) + MaxPool", "17 × 17 × 128", "73,856 params", "36,992"),
        ("GlobalAveragePooling2D", "128", "0 params", "128"),
        ("Dense(64) + Dropout", "64", "8,256 params", "64"),
        ("Dense(1, Sigmoid)", "1", "65 params", "1"),
    ]
    col_headers = ["Capa", "Forma de salida", "Parámetros nuevos", "Activaciones"]
    df_md = "| " + " | ".join(col_headers) + " |\n"
    df_md += "| " + " | ".join(["---"] * 4) + " |\n"
    for row in dims:
        df_md += "| " + " | ".join(row) + " |\n"
    st.markdown(df_md)


# ─────────────────────────────────────────────────────────────────────────────
def sec_casos_uso():
    st.markdown('<div class="section-header"><h2 style="color:#00d4ff;margin:0;">🏭 Casos de Uso en la Industria Real</h2></div>', unsafe_allow_html=True)

    casos = [
        ("🏥", "Medicina y Salud", "#00695C",
         "Detección automatizada de anomalías clínicas",
         ["Diagnóstico de cáncer en mamografías (Google Health — 94.5% vs 88% radiólogo)",
          "Detección de retinopatía diabética en fotografías de fondo de ojo",
          "Segmentación de tumores en MRI y CT",
          "COVID-Net: CNN para detección de COVID-19 en rayos X de tórax"]),
        ("🚗", "Automotriz", "#1565C0",
         "Percepción visual para vehículos autónomos",
         ["Detección de peatones, señales y vehículos en tiempo real (Tesla, Waymo)",
          "Segmentación semántica de la escena de conducción",
          "Detección de obstrucciones y carril de conducción",
          "LIDAR + CNN para navegación en entornos complejos"]),
        ("🔒", "Seguridad", "#E65100",
         "Biometría y reconocimiento facial",
         ["Face ID de Apple: CNN que mapea 30,000 puntos del rostro",
          "Sistemas de videovigilancia con detección de comportamientos anómalos",
          "Verificación de identidad en fronteras aeroportuarias",
          "Deepfake detection: CNNs que identifican rostros sintéticos"]),
        ("📦", "Logística", "#880E4F",
         "Automatización industrial y lectura de documentos",
         ["Amazon Robotics: CNNs para identificar y clasificar paquetes",
          "Lectura automática de códigos QR, de barras y matrícula vehicular",
          "Control de calidad visual en líneas de manufactura (Foxconn, Samsung)",
          "Conteo y clasificación de inventario en tiempo real"]),
        ("🌾", "Agricultura", "#2E7D32",
         "Monitoreo de cultivos con visión artificial",
         ["Detección de enfermedades en plantas desde drones e imágenes satelitales",
          "Clasificación de frutas por madurez y defectos en líneas de empaque",
          "Conteo de plantas y estimación de rendimiento de cosecha",
          "PlantVillage: 99%+ de precisión en 14 enfermedades de cultivos"]),
        ("🎮", "Entretenimiento", "#4A148C",
         "Creación de contenido visual inteligente",
         ["Super-resolución de imágenes y video (DALL-E, Midjourney usan CNNs)",
          "Filtros de realidad aumentada en Instagram/Snapchat (detección facial)",
          "Compresión inteligente de imágenes (JPEG 2000 neural)",
          "Captura de movimiento y animación de personajes 3D"]),
    ]

    cols = st.columns(2)
    for i, (icon, sector, color, app, examples) in enumerate(casos):
        with cols[i % 2]:
            examples_html = "".join(f"<li style='color:#cdd5e0;margin:2px 0;'>{e}</li>" for e in examples)
            st.markdown(f"""
            <div class="concept-card" style="border-left:4px solid {color};">
                <h3 style="color:{color};margin:0 0 4px 0;">{icon} {sector}</h3>
                <p style="color:#8899aa;margin:0 0 8px 0;font-size:0.9em;">{app}</p>
                <ul style="margin:0;padding-left:16px;">{examples_html}</ul>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
def sec_demo():
    st.markdown('<div class="section-header"><h2 style="color:#00d4ff;margin:0;">🧪 Demostración: Clasificación de Rayos X COVID-19</h2></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 Contexto", "🤖 Modelo", "🔬 Predicciones en vivo"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### El problema")
            st.markdown("""
            El diagnóstico de neumonía por COVID-19 mediante radiografías de tórax es crítico
            para el triaje hospitalario. Una CNN puede analizar la imagen en milisegundos
            y detectar patrones de consolidación pulmonar invisible al ojo no entrenado.
            """)
            card("Dataset", "COVID-19 X-Ray Dataset (Kaggle) — 148 imágenes de entrenamiento, 40 de test. Clases balanceadas: NORMAL y PNEUMONIA.")
            card("Objetivo", "Clasificar radiografías de tórax como NORMAL (pulmones sanos) o PNEUMONIA (opacidades y consolidaciones por COVID-19).")

        with c2:
            st.markdown("### CNN desde cero vs Transfer Learning")
            fig, ax = dark_fig(1, 1, figsize=(6, 4))
            metrics = ["Train acc.", "Val acc.", "Test acc.", "Recall Pneum."]
            scratch = [0.63, 0.50, 0.57, 0.15]
            transfer = [0.95, 0.96, 0.97, 1.00]
            x = np.arange(len(metrics))
            w = 0.35
            bars1 = ax.bar(x - w/2, scratch,  w, label="CNN desde cero", color="#f44336", alpha=0.85)
            bars2 = ax.bar(x + w/2, transfer, w, label="MobileNetV2 (TL)", color="#4caf50", alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(metrics, color="#e0e0e0", fontsize=9)
            ax.set_ylim(0, 1.15)
            ax.legend(facecolor="#1a2744", labelcolor="#e0e0e0")
            for bar in bars2:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{bar.get_height():.0%}", ha="center", color="#4caf50", fontsize=8, fontweight="bold")
            for bar in bars1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{bar.get_height():.0%}", ha="center", color="#f44336", fontsize=8, fontweight="bold")
            setup_ax(ax, "CNN desde cero vs Transfer Learning")
            ax.set_ylabel("Métrica")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("⚠️ Recall PNEUMONIA: con CNN desde cero solo detecta 3/20 casos. Transfer Learning detecta 20/20.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Arquitectura: Transfer Learning con MobileNetV2")
            concept_card("🧠", "¿Por qué Transfer Learning?",
                         "Con solo 120 imágenes de entrenamiento, una CNN desde cero no puede aprender filtros útiles. MobileNetV2, preentrenada en <strong>1.2 millones de imágenes</strong> de ImageNet, ya conoce bordes, texturas y formas.")
            concept_card("📦", "MobileNetV2",
                         "Arquitectura de Google diseñada para eficiencia móvil. Usa <em>Inverted Residuals</em> y <em>Linear Bottlenecks</em>. 2.2M parámetros, diseñada para ejecutarse en teléfonos.")
            concept_card("🎯", "Estrategia de dos fases",
                         "<strong>Fase 1:</strong> Congelar la base y entrenar solo la cabeza (1,281 params entrenables) — 15 épocas, lr=1e-3.<br><strong>Fase 2:</strong> Descongelar las últimas 30 capas para fine-tuning — 10 épocas, lr=1e-5.")

        with c2:
            st.markdown("### Flujo del Transfer Learning")
            fig, ax = dark_fig(1, 1, figsize=(5, 7))
            ax.axis("off")
            ax.set_facecolor("#0d1b2a")

            blks = [
                ("Imagen 224×224×3", "#1565C0", 1.0, False),
                ("MobileNetV2 Base\n(2.25M params)\nPreentrenada en ImageNet\n[CONGELADA fase 1]", "#00695C", 0.85, True),
                ("GlobalAvgPool2D\n→ (1280,)", "#2E7D32", 0.35, False),
                ("Dropout(0.3)", "#795548", 0.28, False),
                ("Dense(1, Sigmoid)", "#D84315", 0.28, False),
                ("P(PNEUMONIA)", "#B71C1C", 0.22, False),
            ]

            ys = np.linspace(0.9, 0.08, len(blks))
            for (lbl, col, rel_h, is_base), y in zip(blks, ys):
                h = 0.12 if is_base else 0.07
                r = patches.FancyBboxPatch(
                    (0.1, y - h/2), 0.8, h,
                    boxstyle="round,pad=0.01",
                    facecolor=col, alpha=0.85,
                    transform=ax.transAxes
                )
                ax.add_patch(r)
                ax.text(0.5, y, lbl, transform=ax.transAxes,
                        ha="center", va="center", color="white",
                        fontsize=8, fontweight="bold")

            for i in range(len(blks) - 1):
                ax.annotate("", xy=(0.5, ys[i+1] + 0.038),
                            xytext=(0.5, ys[i] - 0.038),
                            xycoords="axes fraction", textcoords="axes fraction",
                            arrowprops=dict(arrowstyle="->", color="#00d4ff", lw=2))

            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with tab3:
        model = load_model()
        if model is None:
            st.warning("⚠️ Modelo no encontrado. Ejecuta primero el notebook `notebooks/cnn_covid19_xray.ipynb`.")
            return

        if not DATA_DIR.exists():
            st.warning("⚠️ Dataset no encontrado en `data/xray_dataset_covid19/`.")
            return

        import tensorflow as tf

        st.markdown("### Predicciones en tiempo real sobre el conjunto de test")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            selected_cls = st.selectbox("Clase a mostrar:", ["Ambas", "NORMAL", "PNEUMONIA"])
        with c2:
            n_imgs = st.slider("Imágenes a mostrar:", 2, 8, 4, step=2)
        with c3:
            threshold = st.slider("Umbral de clasificación (PNEUMONIA si prob >):", 0.1, 0.9, 0.4, step=0.05)

        classes = ["NORMAL", "PNEUMONIA"]
        if selected_cls == "Ambas":
            files = []
            for cls in classes:
                cls_files = list((DATA_DIR / "test" / cls).glob("*"))
                files.extend(cls_files[:n_imgs // 2])
        else:
            files = list((DATA_DIR / "test" / selected_cls).glob("*"))[:n_imgs]

        if not files:
            st.error("No se encontraron imágenes.")
            return

        random.shuffle(files)
        files = files[:n_imgs]

        results = []
        for fp in files:
            true_cls = fp.parent.name
            img = Image.open(fp).convert("RGB").resize((224, 224))
            arr = np.array(img, dtype=np.float32)
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
            arr = np.expand_dims(arr, 0)
            prob = float(model.predict(arr, verbose=0)[0][0])
            pred = "PNEUMONIA" if prob > threshold else "NORMAL"
            results.append((fp, true_cls, pred, prob, pred == true_cls))

        # Summary metrics
        correct = sum(r[4] for r in results)
        total = len(results)
        fn = sum(1 for r in results if r[1] == "PNEUMONIA" and r[2] == "NORMAL")

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Precisión en muestra", f"{correct}/{total}", f"{correct/total:.0%}")
        mc2.metric("Umbral actual", f"{threshold:.2f}")
        mc3.metric("Falsos negativos (peligro)", fn, delta=f"Neumonía perdida" if fn > 0 else "Ninguno", delta_color="inverse")

        st.divider()
        cols = st.columns(min(4, total))
        for i, (fp, true_cls, pred, prob, ok) in enumerate(results):
            with cols[i % min(4, total)]:
                img_show = Image.open(fp).convert("RGB")
                st.image(img_show, use_container_width=True)
                icon = "✅" if ok else "❌"
                color = "pred-correct" if ok else "pred-incorrect"
                bar_color = "#f44336" if pred == "PNEUMONIA" else "#4caf50"
                st.markdown(f"""
                <div style="text-align:center;padding:4px;">
                    <p style="color:#8899aa;margin:0;font-size:0.8em;">Real: <strong>{true_cls}</strong></p>
                    <p class="{color}" style="margin:2px 0;">{icon} {pred}</p>
                    <div style="background:#2d4070;border-radius:4px;height:8px;margin:4px 0;">
                        <div style="background:{bar_color};width:{prob*100:.0f}%;height:8px;border-radius:4px;"></div>
                    </div>
                    <p style="color:#8899aa;margin:0;font-size:0.8em;">Confianza: {prob:.1%}</p>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
def sec_futuro():
    st.markdown('<div class="section-header"><h2 style="color:#00d4ff;margin:0;">🔮 El Futuro: Más Allá de las CNNs</h2></div>', unsafe_allow_html=True)

    tendencias = [
        ("👁️", "Vision Transformers (ViTs)", "#1565C0",
         "En 2020, Google introdujo ViT: una imagen dividida en parches tratados como tokens por un Transformer. Sin convolución. Con suficientes datos, supera a CNNs en ImageNet.",
         ["Procesamiento global de la imagen desde la primera capa",
          "Mejor escalabilidad con más datos y compute",
          "Menos sesgo inductivo que las CNNs (no asume localidad)",
          "Base de modelos como CLIP, DALL-E, Flamingo"]),
        ("🔀", "Modelos Híbridos CNN+Transformer", "#00695C",
         "La combinación óptima: CNNs para extraer rasgos locales eficientemente, Transformers para integrar contexto global. Lo mejor de ambos mundos.",
         ["ConvNeXt (Meta): CNN moderna con ideas de Transformers",
          "Swin Transformer: ventanas jerárquicas, compite con CNNs",
          "MobileViT: híbrido ultraligero para dispositivos móviles",
          "EfficientFormer: diseñado para latencia mínima en inferencia"]),
        ("📱", "Edge AI y CNNs Ultraligeras", "#E65100",
         "La tendencia es llevar la inteligencia visual a dispositivos con recursos limitados: teléfonos, drones, cámaras de seguridad, sensores IoT.",
         ["MobileNet, EfficientNet-Lite: modelos < 5MB en producción",
          "Cuantización: reducir de float32 a int8 sin perder precisión",
          "Pruning: eliminar conexiones irrelevantes (hasta 90% menos params)",
          "Neural Architecture Search (NAS): diseño automático de arquitecturas"]),
        ("🧬", "CNNs en Ciencias Naturales", "#880E4F",
         "Las CNNs están revolucionando disciplinas que trabajan con datos espaciales: astronomía, biología molecular, sismología y física de partículas.",
         ["AlphaFold2 usa atención sobre estructuras 3D de proteínas",
          "Detección de exoplanetas en curvas de luz del telescopio Kepler",
          "Clasificación de galaxias por morfología a escala de millones",
          "Identificación de señales de ondas gravitacionales (LIGO)"]),
    ]

    c1, c2 = st.columns(2)
    for i, (icon, title, color, desc, bullets) in enumerate(tendencias):
        with (c1 if i % 2 == 0 else c2):
            bullets_html = "".join(f"<li style='color:#cdd5e0;margin:3px 0;font-size:0.9em;'>{b}</li>" for b in bullets)
            st.markdown(f"""
            <div class="concept-card" style="border-left:4px solid {color};">
                <h3 style="color:{color};margin:0 0 6px 0;">{icon} {title}</h3>
                <p style="color:#8899aa;margin:0 0 10px 0;font-size:0.9em;">{desc}</p>
                <ul style="margin:0;padding-left:16px;">{bullets_html}</ul>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Bibliografía")
    refs = [
        "LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.",
        "Wang, L., & Wong, A. (2020). COVID-Net: A tailored deep convolutional neural network design for detection of COVID-19 cases from chest X-ray images. *Scientific Reports*, 10, 19549.",
        "Sandler, M. et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR 2018*.",
        "Dosovitskiy, A. et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.",
        "Khoong, W.H. (2020). COVID-19 X-Ray Dataset. Kaggle. `khoongweihao/covid19-xray-dataset-train-test-sets`",
        "Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.",
    ]
    for i, r in enumerate(refs, 1):
        st.markdown(f"{i}. {r}")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 CNN — Analítica de Datos")
    st.markdown("*Universidad de Antioquia*")
    st.divider()
    section = st.radio(
        "Navegación",
        [
            "🏠 Inicio",
            "❓ ¿Por qué no MLP?",
            "📅 Evolución Histórica",
            "⚙️ Capa Convolucional",
            "⚡ Función de Activación",
            "🗜️ Pooling",
            "🔗 Capas Densas",
            "🔄 Flujo Completo",
            "🏭 Casos de Uso",
            "🧪 Demo: COVID-19",
            "🔮 Futuro",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Profesor:** Duván Cataño")
    st.markdown("**Dataset:** COVID-19 X-Ray")
    st.caption("Tópicos de Machine Learning")

# ── ROUTING ───────────────────────────────────────────────────────────────────
{
    "🏠 Inicio":              sec_inicio,
    "❓ ¿Por qué no MLP?":   sec_por_que_no_mlp,
    "📅 Evolución Histórica": sec_historia,
    "⚙️ Capa Convolucional": sec_convolucion,
    "⚡ Función de Activación": sec_activacion,
    "🗜️ Pooling":             sec_pooling,
    "🔗 Capas Densas":        sec_capas_densas,
    "🔄 Flujo Completo":      sec_flujo_completo,
    "🏭 Casos de Uso":        sec_casos_uso,
    "🧪 Demo: COVID-19":      sec_demo,
    "🔮 Futuro":              sec_futuro,
}[section]()
