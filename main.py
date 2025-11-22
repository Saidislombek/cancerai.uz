import time
from pathlib import Path

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
import gdown  # для скачивания модели с Google Drive


# =========================================================
#     ПУТИ К ФАЙЛУ МОДЕЛИ И ССЫЛКА НА GOOGLE DRIVE
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# Папка и путь к локальному файлу модели (на Streamlit Cloud тоже)
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "cc_vit_sts.h5"

# Ссылка на файл .h5 в Google Drive:
# https://drive.google.com/file/d/1vzqeIPnuUTdFRaqjfXYaxXxMX-LpFyKC/view?usp=sharing
DEFAULT_MODEL_URL = (
    "https://drive.google.com/uc"
    "?export=download&id=1vzqeIPnuUTdFRaqjfXYaxXxMX-LpFyKC"
)

# Позволяем переопределить URL через secrets (если захочешь)
if "MODEL_URL" in st.secrets:
    MODEL_URL = st.secrets["MODEL_URL"]
else:
    MODEL_URL = DEFAULT_MODEL_URL

IMAGE_SIZE = 224  # входной размер для Swin Small


def ensure_model_file() -> None:
    """
    Проверяет наличие файла модели локально.
    Если файла нет — скачивает его из Google Drive по MODEL_URL.
    """
    if MODEL_PATH.exists():
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваем модель из Google Drive в {MODEL_PATH}...")

    try:
        gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)
    except Exception as e:
        # Ошибка скачивания — прерываем работу приложения
        raise RuntimeError(f"Не удалось скачать файл модели: {e}") from e

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Файл модели не был скачан. Проверьте MODEL_URL.")


# =========================================================
#     НАСТРОЙКА СТРАНИЦЫ + CSS
# =========================================================

st.set_page_config(
    page_title="CancerAI - Диагностика рака шейки матки",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>

    .st-emotion-cache-zy6yx3 {
         padding: 30px 0px !important;
    }

    .stApp {
        background-color: #ffffff !important;
    }

    /* Общий контейнер по центру страницы */
    .page-container {
        max-width: 820px;
        margin: 0px auto;
        padding: 0px;
    }

    /* На всякий случай центрируем h3/h4 внутри контейнера */
    .page-container h3,
    .page-container h4 {
        text-align: center;
    }

    .result-title {
        font-size: 28px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 4px;
    }

    .result-subtitle {
        font-size: 18px;
        font-weight: 600;
        color: #6b7280;
        text-align: center;
        margin-bottom: 18px;
    }

    table.metrics-table,
    table.classes-table {
        border-collapse: collapse;
        width: 600px;
        max-width: 600px;
        margin-top: 8px;
        margin-left: auto;
        margin-right: auto;
    }

    table.metrics-table th,
    table.metrics-table td,
    table.classes-table th,
    table.classes-table td {
        border: 2px solid #000000;
        padding: 6px 10px;
        font-size: 16px;
        text-align: center;
    }

    table.metrics-table th,
    table.classes-table th {
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
    st.markdown("### ⚙️ Сервисные операции")
    if st.button("🧹 Очистить кэш модели"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Кэш очищен. Модель будет загружена заново при следующем прогнозе.")


# =========================================================
#     ЗАГРУЗКА МОДЕЛИ
# =========================================================

@st.cache_resource
def load_model_and_meta():
    """
    Загружает архитектуру Swin-S и веса из файла cc_vit_sts.h5.
    Если файл модели отсутствует, сначала скачивает его по MODEL_URL.
    """
    # 1. Убедиться, что файл модели есть (если нет — скачать)
    ensure_model_file()

    # 2. Открываем уже гарантированно существующий файл
    with h5py.File(MODEL_PATH, "r") as f:
        attrs = dict(f["info"].attrs)

        class_names = attrs["classes"].split(",")  # HSIL,LSIL,NILM,SCC
        model_name = attrs["model_name"]           # swin_small_patch4_window7_224

        state = {}
        for k in f["model_state_dict"].keys():
            np_arr = f["model_state_dict"][k][()]
            state[k] = torch.from_numpy(np_arr)

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(class_names),
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, class_names


# =========================================================
#     ПРЕДОБРАБОТКА + ПРОГНОЗ
# =========================================================

def preprocess(img: Image.Image) -> torch.Tensor:
    """
    Преобразование изображения к формату, ожидаемому моделью:
    resize -> tensor -> нормализация.
    """
    tfm = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    return tfm(img.convert("RGB")).unsqueeze(0)


def predict_single(img: Image.Image):
    """
    Прогноз по одному изображению.
    """
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

# Весь контент страницы в одном центральном контейнере
st.markdown('<div class="page-container">', unsafe_allow_html=True)

# Заголовок и описание по центру
st.markdown(
    "<h2 style='text-align:center;'>🧬 Классификация фенотипов рака шейки матки</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align:center; color:#6b7280;'>"
    "Загрузите цитологическое изображение.<br>Модель Swin-S выполнит прогноз фенотипа рака шейки матки."
    "</h4>",
    unsafe_allow_html=True,
)

# Блок загрузки и кнопка — по центру, через колонки
col_u1, col_u2, col_u3 = st.columns([1, 2, 1])

with col_u2:
    st.markdown("<h4>Загрузите изображение (JPG/PNG)</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png"],
        help="Выберите цитологическое изображение для анализа.",
    )
    btn = st.button("🔍 Выполнить прогноз")

# Логика обработки
if btn:
    if uploaded_file is None:
        st.warning("Пожалуйста, сначала загрузите изображение.")
    else:
        image = Image.open(uploaded_file)

        # Сразу считаем прогноз
        with st.spinner("Модель выполняет прогноз..."):
            pred_class, confidence, probs, elapsed, class_names = predict_single(image)

        elapsed_s = f"{elapsed:.3f} сек"
        conf_s = f"{confidence * 100:.2f} %"

        # --------------------------------------------
        # БЛОК РЕЗУЛЬТАТОВ
        # --------------------------------------------
        st.markdown('<div class="page-container">', unsafe_allow_html=True)

        st.markdown(
            '<div class="result-title">📊 Результаты диагностики</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="result-subtitle">Результаты, проанализированные моделью искусственного интеллекта</div>',
            unsafe_allow_html=True,
        )

        # ---------- 1. ИТОГОВЫЕ ПОКАЗАТЕЛИ ----------
        st.markdown(
            "<h3 style='text-align:center;'>Итоговые показатели</h3>",
            unsafe_allow_html=True,
        )

        metrics_names = [
            "Время на прогноз",
            "Точность прогнозирования",
            "Предсказанный класс",
        ]
        metrics_values = [elapsed_s, conf_s, pred_class]

        df_metrics = pd.DataFrame(
            {
                "№": list(range(1, len(metrics_names) + 1)),  # нумерация с 1
                "Показатель": metrics_names,
                "Значение": metrics_values,
            }
        )

        metrics_html = df_metrics.to_html(
            index=False,
            classes="metrics-table",
            border=0,
            escape=False,
        )
        st.markdown(metrics_html, unsafe_allow_html=True)

        # ---------- 2. ДЕТАЛИЗАЦИЯ ПО ВСЕМ КЛАССАМ ----------
        st.markdown(
            "<h3 style='text-align:center;'>Детализация по всем классам</h3>",
            unsafe_allow_html=True,
        )

        df_classes = pd.DataFrame(
            {
                "№": list(range(len(class_names))),  # 0,1,2,...
                "Класс": class_names,
                "Вероятность, %": [round(float(p) * 100, 2) for p in probs],
            }
        )

        classes_html = df_classes.to_html(
            index=False,
            classes="classes-table",
            border=0,
            escape=False,
        )
        st.markdown(classes_html, unsafe_allow_html=True)

        # ---------- 3. ЗАГРУЖЕННОЕ ИЗОБРАЖЕНИЕ (ПО ЦЕНТРУ) ----------
        st.markdown(
            "<h3 style='text-align:center;'>Загруженное изображение</h3>",
            unsafe_allow_html=True,
        )

        img_left, img_center, img_right = st.columns([1, 2, 1])
        with img_center:
            st.image(image, width=700)

        # Закрываем внутренний .page-container (блок результатов)
        st.markdown("</div>", unsafe_allow_html=True)

# Закрываем внешний .page-container
st.markdown("</div>", unsafe_allow_html=True)
