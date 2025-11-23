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
MODEL_URL = st.secrets.get("MODEL_URL", DEFAULT_MODEL_URL)

IMAGE_SIZE = 224  # входной размер для Swin Small


def _download_model() -> None:
    """Качает модель из Google Drive в MODEL_PATH."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваем модель из Google Drive в {MODEL_PATH}...")
    gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)


def ensure_model_file(force: bool = False) -> None:
    """
    Гарантирует, что локальный файл модели существует и является
    корректным HDF5. Если файла нет или он битый — перекачивает.

    force=True можно использовать для принудительного перескачивания.
    """
    # Если явно попросили — удаляем старый файл
    if force and MODEL_PATH.exists():
        MODEL_PATH.unlink()

    # Если файла нет — качаем
    if not MODEL_PATH.exists():
        _download_model()

    # Проверяем, что файл действительно HDF5, а не HTML/обрезок
    try:
        with h5py.File(MODEL_PATH, "r") as f:
            _ = list(f.keys())  # просто чтение структуры
    except OSError:
        # Первый запуск: файл битый или не HDF5 — пробуем перекачать
        print("Файл модели повреждён или не является HDF5. Перекачиваем...")
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        _download_model()

        # Вторая попытка; если снова ошибка — падаем с понятным текстом
        try:
            with h5py.File(MODEL_PATH, "r") as f:
                _ = list(f.keys())
        except OSError as e2:
            raise RuntimeError(
                "Не удалось открыть скачанный файл модели как HDF5. "
                "Проверь, что файл в Google Drive именно .h5 и доступен "
                "'Anyone with the link'."
            ) from e2


# =========================================================
#     НАСТРОЙКА СТРАНИЦЫ + CSS
# =========================================================

st.set_page_config(
    page_title="CancerAI - Диагностика рака шейки матки",
    page_icon="🧬",
    layout="wide",
)

HIDE_STREAMLIT_STYLE = """
<style>
/* Скрыть стандартное меню Streamlit */
#MainMenu {
    visibility: hidden;
}

/* Скрыть верхний и нижний бар приложения */
header {
    visibility: hidden;
}
footer {
    visibility: hidden;
}
</style>
"""

st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Глобально говорим браузеру, что страница светлая */
    :root {
        color-scheme: light;
    }

    /* Основной контейнер приложения: белый фон + тёмный текст */
    .stApp {
        background-color: #ffffff !important;
        color: #111827 !important;
    }

    /* Тёмный текст для всех базовых элементов */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp span, .stApp label, .stApp li, .stApp div {
        color: #111827;
    }

    /* Сайдбар (светлый, как в light-теме) */
    [data-testid="stSidebar"] {
        background-color: #f9fafb !important;
        color: #111827 !important;
        border-right: 1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] * {
        color: #111827 !important;
    }

    /* Кнопки (все: и в сайдбаре, и в основном контенте) */
    .stButton > button {
        background-color: #0f766e !important;      /* primary */
        color: #ffffff !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 0.40rem 1.2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.25);
        transition: background-color 0.15s ease, transform 0.08s ease,
                    box-shadow 0.15s ease;
    }

    .stButton > button:hover {
        background-color: #0b524c !important;
        box-shadow: 0 8px 18px rgba(15, 118, 110, 0.35);
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 3px 8px rgba(15, 118, 110, 0.20);
    }

    /* Файл-загрузчик: светлый бокс */
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

    [data-testid="stFileUploader"] label {
        color: #4b5563 !important;
        font-weight: 500;
    }

    /* Кнопка Browse files внутри загрузчика */
    [data-testid="stFileUploader"] button {
        background-color: #0f766e !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 0.30rem 0.9rem !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        box-shadow: 0 3px 8px rgba(15, 118, 110, 0.25);
        transition: background-color 0.15s ease, transform 0.08s ease,
                    box-shadow 0.15s ease;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #0b524c !important;
        box-shadow: 0 6px 14px rgba(15, 118, 110, 0.35);
        transform: translateY(-1px);
    }

    [data-testid="stFileUploader"] button:active {
        transform: translateY(0);
        box-shadow: 0 3px 8px rgba(15, 118, 110, 0.20);
    }

    /* Обнуляем лишний вертикальный padding */
    .st-emotion-cache-zy6yx3 {
         padding: 30px 0px !important;
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
        # принудительно удалим и файл, чтобы точно перекачался
        ensure_model_file(force=True)
        st.success(
            "Кэш и файл модели очищены. "
            "Модель будет загружена заново при следующем прогнозе."
        )


# =========================================================
#     ЗАГРУЗКА МОДЕЛИ
# =========================================================

@st.cache_resource
def load_model_and_meta():
    """
    Загружает архитектуру Swin-S и веса из файла cc_vit_sts.h5.
    Если файл модели отсутствует или битый, ensure_model_file()
    его скачает/перекачает.
    """
    ensure_model_file()

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
    "Загрузите цитологическое изображение.<br>"
    "Модель Swin-S выполнит прогноз фенотипа рака шейки матки."
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
            '<div class="result-subtitle">'
            "Результаты, проанализированные моделью искусственного интеллекта"
            "</div>",
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

# =========================
#     КАСТОМНЫЙ ФУТЕР
# =========================

FOOTER_HTML = """
<style>
.cai-footer {
    background: #020617;          /* очень тёмный фон */
    color: #e5e7eb;
    padding: 48px 0 28px 0;
    margin-top: 72px;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                 sans-serif;
}

/* контейнер по центру */
.cai-footer__container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 24px;
    display: flex;
    flex-direction: column;
    gap: 32px;
}

/* верхняя часть футера: логотип + колонки */
.cai-footer__top {
    display: flex;
    flex-direction: column;
    gap: 32px;
}

@media (min-width: 900px) {
    .cai-footer__top {
        flex-direction: row;
        justify-content: space-between;
        align-items: flex-start;
    }
}

/* блок логотипа / описания */
.cai-footer__brand {
    display: flex;
    flex-direction: column;
    gap: 18px;
    max-width: 420px;
}

.cai-footer__logo-row {
    display: flex;
    align-items: center;
    gap: 14px;
}

.cai-footer__logo-circle {
    width: 44px;
    height: 44px;
    border-radius: 999px;
    background: #22c55e22;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}

.cai-footer__brand-name {
    font-size: 24px;
    font-weight: 700;
}

.cai-footer__tagline {
    font-size: 14px;
    line-height: 1.5;
    color: #9ca3af;
}

/* социальные кнопки */
.cai-footer__socials {
    display: flex;
    gap: 12px;
}

.cai-footer__social {
    width: 34px;
    height: 34px;
    border-radius: 999px;
    background: #020617;
    border: 1px solid #374151;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    text-decoration: none;
    color: #e5e7eb;
    transition: background 0.15s ease, transform 0.08s ease,
                border-color 0.15s ease;
}

.cai-footer__social:hover {
    background: #111827;
    border-color: #4b5563;
    transform: translateY(-1px);
}

/* колонки ссылок */
.cai-footer__cols {
    display: flex;
    flex-direction: row;
    gap: 40px;
    flex-wrap: wrap;
}

.cai-footer__col-title {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .03em;
    color: #9ca3af;
    margin-bottom: 10px;
}

.cai-footer__link {
    display: block;
    font-size: 14px;
    color: #e5e7eb;
    text-decoration: none;
    margin-bottom: 6px;
    transition: color 0.15s ease;
}

.cai-footer__link:hover {
    color: #22c55e;
}

/* разделитель и нижняя строка */
.cai-footer__divider {
    border-top: 1px solid #111827;
    margin-top: 8px;
    padding-top: 16px;
}

.cai-footer__bottom {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 13px;
    color: #9ca3af;
}

@media (min-width: 700px) {
    .cai-footer__bottom {
        flex-direction: row;
        justify-content: space-between;
        align-items: center;
    }
}

.cai-footer__author {
    color: #22c55e;
}
</style>

<div class="cai-footer">
  <div class="cai-footer__container">

    <div class="cai-footer__top">

      <!-- Левая часть: логотип + текст + соцсети -->
      <div class="cai-footer__brand">
        <div class="cai-footer__logo-row">
          <div class="cai-footer__logo-circle">🧬</div>
          <div class="cai-footer__brand-name">CancerAI</div>
        </div>
        <div class="cai-footer__tagline">
          AI-система для классификации цитологических изображений
          и прогнозирования фенотипов рака шейки матки.
        </div>
        <div class="cai-footer__socials">
          <!-- сюда подставь свои реальные ссылки -->
          <a class="cai-footer__social" href="https://t.me/your_telegram" target="_blank" rel="noopener">📨</a>
          <a class="cai-footer__social" href="https://instagram.com/your_instagram" target="_blank" rel="noopener">📸</a>
          <a class="cai-footer__social" href="https://github.com/Saidislombek" target="_blank" rel="noopener">🐱</a>
        </div>
      </div>

      <!-- Правая часть: колонки -->
      <div class="cai-footer__cols">
        <div>
          <div class="cai-footer__col-title">Сервис</div>
          <a class="cai-footer__link" href="#upload">Классификация снимка</a>
          <a class="cai-footer__link" href="#">Руководство по использованию</a>
          <a class="cai-footer__link" href="#">Ограничения модели</a>
        </div>

        <div>
          <div class="cai-footer__col-title">Проект</div>
          <a class="cai-footer__link" href="#">О CancerAI</a>
          <a class="cai-footer__link" href="#">Контакты</a>
          <a class="cai-footer__link" href="#">Политика конфиденциальности</a>
        </div>
      </div>

    </div>

    <div class="cai-footer__divider">
      <div class="cai-footer__bottom">
        <span>© 2025 CancerAI. Все права защищены.</span>
        <span>Создано <span class="cai-footer__author">
          Abdullakhujaev Saidislombek N.
        </span></span>
      </div>
    </div>

  </div>
</div>
"""
st.markdown(FOOTER_HTML, unsafe_allow_html=True)
