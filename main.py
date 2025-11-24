import time
from pathlib import Path

import gdown
import h5py
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import torch
import torch.nn.functional as F
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


# =========================================================
#     ПУТИ К ФАЙЛУ МОДЕЛИ И ССЫЛКА НА GOOGLE DRIVE
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "cc_vit_sts.h5"

DEFAULT_MODEL_URL = (
    "https://drive.google.com/uc"
    "?export=download&id=1vzqeIPnuUTdFRaqjfXYaxXxMX-LpFyKC"
)

MODEL_URL = st.secrets.get("MODEL_URL", DEFAULT_MODEL_URL)

IMAGE_SIZE = 224  # входной размер для Swin Small


# =========================================================
#     РАБОТА С ФАЙЛОМ МОДЕЛИ
# =========================================================

def _download_model() -> None:
    """Качает модель из Google Drive в MODEL_PATH."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваем модель из Google Drive в {MODEL_PATH}...")
    gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)


def ensure_model_file(force: bool = False) -> None:
    if force and MODEL_PATH.exists():
        MODEL_PATH.unlink()

    if not MODEL_PATH.exists():
        _download_model()

    try:
        with h5py.File(MODEL_PATH, "r") as f:
            _ = list(f.keys())
    except OSError:
        print("Файл модели повреждён или не является HDF5. Перекачиваем...")
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        _download_model()


# =========================================================
#     НАСТРОЙКА СТРАНИЦЫ + CSS (ГЛАВНОЕ ИСПРАВЛЕНИЕ ЗДЕСЬ)
# =========================================================

st.set_page_config(
    page_title="CancerAI - Диагностика рака шейки матки",
    page_icon="DNA",
    layout="wide",
)

HIDE_STREAMLIT_STYLE = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {display: none !important;}
</style>
"""
st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    :root {color-scheme: light;}

    .stApp {
        background-color: #ffffff !important;
        color: #111827 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #f9fafb !important;
        border-right: 1px solid #e5e7eb;
    }

    /* ==================== КНОПКИ — БЕЛЫЙ ТЕКСТ ВЕЗДЕ ==================== */
    .stButton > button {
        background-color: #0f766e !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 0.40rem 1.2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.25);
        transition: all 0.2s ease;
    }

    /* Принудительно белый текст — даже если Streamlit переопределяет */
    .stButton > button,
    .stButton > button span,
    .stButton > button * {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        background-color: #0b524c !important;
        color: #ffffff !important;
        box-shadow: 0 8px 18px rgba(15, 118, 110, 0.35) !important;
        transform: translateY(-1px);
    }

    .stButton > button:active {
        color: #ffffff !important;
        transform: translateY(0);
        box-shadow: 0 3px 8px rgba(15, 118, 110, 0.20);
    }

    /* Кнопка "Browse files" внутри file_uploader */
    [data-testid="stFileUploader"] button {
        background-color: #0f766e !important;
        color: #ffffff !important;
        border-radius: 9999px !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #0b524c !important;
    }

    /* Абсолютная страховка — если Streamlit опять поменяет атрибуты */
    button[kind="primary"],
    button[kind="secondary"] {
        color: #ffffff !important;
    }

    /* Остальные стили */
    [data-testid="stFileUploader"] > section {
        border-radius: 12px;
        border: 2px dashed #d1d5db;
        background-color: #f9fafb;
        padding: 1.25rem;
    }

    [data-testid="stFileUploader"] > section:hover {
        border-color: #0f766e;
        background-color: #f3f4ff;
    }

    .page-container {
        max-width: 820px;
        margin: 0 auto;
        padding: 0;
    }

    .result-title {font-size: 28px; font-weight: 700; text-align: center; margin-bottom: 4px;}
    .result-subtitle {font-size: 18px; font-weight: 600; color: #6b7280; text-align: center; margin-bottom: 18px;}

    table.metrics-table, table.classes-table {
        border-collapse: collapse;
        width: 600px;
        max-width: 600px;
        margin: 20px auto;
        border: 2px solid #000;
    }

    table.metrics-table th, table.metrics-table td,
    table.classes-table th, table.classes-table td {
        border: 2px solid #000;
        padding: 8px 12px;
        text-align: center;
        font-size: 16px;
    }

    table.metrics-table th, table.classes-table th {
        background-color: #f9fafb;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
#     САЙДБАР: ОЧИСТКА КЭША
# =========================================================

with st.sidebar:
    st.markdown("### Settings Сервисные операции")
    if st.button("Очистить кэш модели"):
        st.cache_data.clear()
        st.cache_resource.clear()
        ensure_model_file(force=True)
        st.success("Кэш и файл модели очищены. Модель будет загружена заново.")


# =========================================================
#     ЗАГРУЗКА МОДЕЛИ
# =========================================================

@st.cache_resource
def load_model_and_meta():
    ensure_model_file()

    with h5py.File(MODEL_PATH, "r") as f:
        attrs = dict(f["info"].attrs)
        class_names = attrs["classes"].split(",")
        model_name = attrs["model_name"]

        state = {}
        for k in f["model_state_dict"].keys():
            state[k] = torch.from_numpy(f["model_state_dict"][k][()])

    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, class_names


# =========================================================
#     ПРЕДОБРАБОТКА + ПРОГНОЗ
# =========================================================

def preprocess(img: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return tfm(img.convert("RGB")).unsqueeze(0)


def predict_single(img: Image.Image):
    model, class_names = load_model_and_meta()
    x = preprocess(img)

    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(x)
        elapsed = time.perf_counter() - t0

        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        pred_class = class_names[idx]

    return pred_class, confidence, probs, elapsed, class_names


# =========================================================
#     UI
# =========================================================

st.markdown('<div class="page-container">', unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>Классификация фенотипов рака шейки матки</h2>", unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align:center; color:#6b7280;'>"
    "Загрузите цитологическое изображение.<br>"
    "Модель Swin-S выполнит прогноз фенотипа рака шейки матки."
    "</h4>",
    unsafe_allow_html=True,
)

col_u1, col_u2, col_u3 = st.columns([1, 2, 1])
with col_u2:
    st.markdown("<h4>Загрузите изображение (JPG/PNG)</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png"],
        help="Выберите цитологическое изображение для анализа.",
        label_visibility="collapsed"
    )
    btn = st.button("Выполнить прогноз")

if btn:
    if uploaded_file is None:
        st.warning("Пожалуйста, сначала загрузите изображение.")
    else:
        image = Image.open(uploaded_file)

        with st.spinner("Модель выполняет прогноз..."):
            pred_class, confidence, probs, elapsed, class_names = predict_single(image)

        elapsed_s = f"{elapsed:.3f} сек"
        conf_s = f"{confidence * 100:.2f} %"

        st.markdown("<h3 style='text-align:center;'>Итоговые показатели</h3>", unsafe_allow_html=True)

        df_metrics = pd.DataFrame({
            "№": [1, 2, 3],
            "Показатель": ["Время на прогноз", "Точность прогнозирования", "Предсказанный класс"],
            "Значение": [elapsed_s, conf_s, pred_class]
        })
        st.markdown(df_metrics.to_html(classes="metrics-table", index=False, border=0), unsafe_allow_html=True)

        st.markdown("<h3 style='text-align:center;'>Детализация по всем классам</h3>", unsafe_allow_html=True)

        df_classes = pd.DataFrame({
            "№": list(range(len(class_names))),
            "Класс": class_names,
            "Вероятность, %": [round(float(p) * 100, 2) for p in probs],
        })
        st.markdown(df_classes.to_html(classes="classes-table", index=False, border=0), unsafe_allow_html=True)

        st.markdown("<h3 style='text-align:center;'>Загруженное изображение</h3>", unsafe_allow_html=True)
        img_l, img_c, img_r = st.columns([1, 2, 1])
        with img_c:
            st.image(image, width=700)

st.markdown("</div>", unsafe_allow_html=True)
