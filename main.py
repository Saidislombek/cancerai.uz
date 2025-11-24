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
#     ПУТИ К МОДЕЛИ
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "cc_vit_sts.h5"

DEFAULT_MODEL_URL = "https://drive.google.com/uc?export=download&id=1vzqeIPnuUTdFRaqjfXYaxXxMX-LpFyKC"
MODEL_URL = st.secrets.get("MODEL_URL", DEFAULT_MODEL_URL)
IMAGE_SIZE = 224

def _download_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)

def ensure_model_file(force=False):
    if force and MODEL_PATH.exists():
        MODEL_PATH.unlink()
    if not MODEL_PATH.exists():
        _download_model()
    try:
        with h5py.File(MODEL_PATH, "r"): pass
    except:
        MODEL_PATH.unlink()
        _download_model()

# =========================================================
#     СТИЛИ — ГЛАВНОЕ ИСПРАВЛЕНИЕ ЗДЕСЬ
# =========================================================

st.set_page_config(page_title="CancerAI", page_icon="DNA", layout="wide")

st.markdown("""
<style>
/* Скрываем мусор Streamlit */
#MainMenu, header, footer {visibility: hidden;}
[data-testid="collapsedControl"] {display: none !important;}

/* Фон и цвета */
.stApp {background: white;}
[data-testid="stSidebar"] {background: #f9fafb;}

/* ВСЕ КНОПКИ — БЕЛЫЙ ТЕКСТ НАВСЕГДА */
button[kind="primary"], button[kind="secondary"], .stButton > button {
    background-color: #0f766e !important;
    color: white !important;
    border: none !important;
    border-radius: 9999px !important;
    font-weight: 600 !important;
}

/* Перебиваем даже самые упрямые стили Streamlit */
button[kind="primary"] *, button[kind="secondary"] *, .stButton > button * {
    color: white !important;
}

/* Наведение */
button:hover, .stButton > button:hover {
    background-color: #0b524c !important;
    color: white !important;
}

/* Специально для кнопки "Browse files" в file uploader */
div[data-testid="stFileUploader"] button {
    background-color: #0f766e !important;
    color: white !important;
    border-radius: 9999px !important;
}
div[data-testid="stFileUploader"] button * {
    color: white !important;
}
div[data-testid="stFileUploader"] button:hover {
    background-color: #0b524c !important;
}

/* Остальные стили */
[data-testid="stFileUploader"] > section {
    border: 2px dashed #d1d5db;
    border-radius: 12px;
    background: #f9fafb;
    padding: 1.25rem;
}
[data-testid="stFileUploader"] > section:hover {
    border-color: #0f766e;
    background: #f3f4ff;
}

.page-container {max-width: 820px; margin: 0 auto;}
table.metrics-table, table.classes-table {
    width: 600px; max-width: 100%; margin: 20px auto;
    border-collapse: collapse; border: 2px solid #000;
}
table.metrics-table th, table.metrics-table td,
table.classes-table th, table.classes-table td {
    border: 2px solid #000; padding: 10px; text-align: center;
}
table th {background: #f9fafb; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# =========================================================
#     САЙДБАР
# =========================================================

with st.sidebar:
    st.markdown("### Сервисные операции")
    if st.button("Очистить кэш модели"):
        st.cache_data.clear()
        st.cache_resource.clear()
        ensure_model_file(force=True)
        st.success("Кэш очищен. Модель перезагружена.")

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
        state = {k: torch.from_numpy(f["model_state_dict"][k][()]) for k in f["model_state_dict"].keys()}
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, class_names

# =========================================================
#     ИНФЕРЕНС
# =========================================================

def preprocess(img):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return tfm(img.convert("RGB")).unsqueeze(0)

def predict_single(img):
    model, class_names = load_model_and_meta()
    x = preprocess(img)
    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(x)
        elapsed = time.perf_counter() - t0
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        idx = np.argmax(probs)
        return class_names[idx], float(probs[idx]), probs, elapsed, class_names

# =========================================================
#     UI
# =========================================================

st.markdown('<div class="page-container">', unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>Классификация фенотипов рака шейки матки</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#6b7280;'>Загрузите цитологическое изображение.<br>Модель Swin-S выполнит прогноз фенотипа рака шейки матки.</h4>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    btn = st.button("Выполнить прогноз")

if btn:
    if not uploaded_file:
        st.warning("Сначала загрузите изображение!")
    else:
        image = Image.open(uploaded_file)
        with st.spinner("Анализ..."):
            pred_class, confidence, probs, elapsed, class_names = predict_single(image)

        st.markdown("### Итоговые показатели")
        df1 = pd.DataFrame({
            "№": [1, 2, 3],
            "Показатель": ["Время на прогноз", "Точность прогнозирования", "Предсказанный класс"],
            "Значение": [f"{elapsed:.3f} сек", f"{confidence*100:.2f}%", pred_class]
        })
        st.markdown(df1.to_html(classes="metrics-table", index=False), unsafe_allow_html=True)

        st.markdown("### Детализация по всем классам")
        df2 = pd.DataFrame({
            "№": range(len(class_names)),
            "Класс": class_names,
            "Вероятность, %": [f"{p*100:.2f}" for p in probs]
        })
        st.markdown(df2.to_html(classes="classes-table", index=False), unsafe_allow_html=True)

        st.markdown("### Загруженное изображение")
        cc1, cc2, cc3 = st.columns([1, 2, 1])
        with cc2:
            st.image(image, use_column_width=True)

st.markdown("</div>", unsafe_allow_html=True)
