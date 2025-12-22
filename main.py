# -*- coding: utf-8 -*-
"""
CancerAI (cancerai.uz) — Streamlit inference app.

Key design goals for Railway:
1) The app can be started with `python main.py` (no `streamlit run` command required).
2) The service binds to Railway's provided PORT and 0.0.0.0.
3) The ML model is downloaded at most once per container start and then cached in memory.
4) All UI text is in English.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn.functional as F
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Optional (used when downloading from Google Drive)
try:
    import gdown  # type: ignore
except Exception:  # pragma: no cover
    gdown = None


# =========================================================
# Configuration
# =========================================================

APP_TITLE = "CancerAI — Cervical Cytology Classification"
APP_ICON = "🧬"

# Class order used in your dataset/project
CLASS_NAMES: List[str] = ["HSIL", "LSIL", "NILM", "SCC"]

CLASS_DESCRIPTIONS: Dict[str, str] = {
    "NILM": (
        "Negative for intraepithelial lesion or malignancy. "
        "Cells look within normal limits."
    ),
    "LSIL": (
        "Low-grade squamous intraepithelial lesion. "
        "Typically reflects mild dysplasia / HPV-related changes."
    ),
    "HSIL": (
        "High-grade squamous intraepithelial lesion. "
        "Suggests moderate to severe dysplasia and requires clinical follow-up."
    ),
    "SCC": (
        "Squamous cell carcinoma. "
        "This category indicates malignant changes and needs urgent clinical review."
    ),
}

DISCLAIMER = (
    "Disclaimer: This tool is for research/educational purposes and does not provide medical advice. "
    "Always consult qualified healthcare professionals for diagnosis and treatment decisions."
)

DEFAULT_MODEL_FILENAME = "cc_vit_sts.h5"

# If you mount a Railway Volume, set MODEL_DIR to a persistent path, e.g. /data/models
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models")).resolve()
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / DEFAULT_MODEL_FILENAME))).resolve()

# If MODEL_PATH does not exist, the app will try to download the model from MODEL_URL.
# Example: https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
MODEL_URL = os.getenv("MODEL_URL", "").strip()

# Timm model name must match your training architecture.
TIMM_MODEL_NAME = os.getenv("TIMM_MODEL_NAME", "swin_small_patch4_window7_224")

NUM_CLASSES = int(os.getenv("NUM_CLASSES", str(len(CLASS_NAMES))))

DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "60"))
DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", "3"))


# =========================================================
# Helpers: Streamlit runtime detection and server bootstrap
# =========================================================

def _in_streamlit_runtime() -> bool:
    """
    Returns True when the script is being executed by Streamlit.
    This prevents infinite recursion when we bootstrap Streamlit from `python main.py`.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        return get_script_run_ctx() is not None
    except Exception:
        return any(k.startswith("STREAMLIT_") for k in os.environ.keys())


def _bootstrap_streamlit_server() -> None:
    """
    Start a Streamlit server programmatically so `python main.py` works on Railway.
    """
    port = int(os.getenv("PORT", "8501"))

    flag_options = {
        "server.headless": True,
        "server.address": "0.0.0.0",
        "server.port": port,
        "browser.gatherUsageStats": False,
        "server.enableCORS": False,
        "server.enableXsrfProtection": False,
    }

    main_script = str(Path(__file__).resolve())

    from streamlit.web import bootstrap  # type: ignore

    # Support multiple Streamlit versions (signature changed across releases).
    try:
        bootstrap.run(main_script, args=[], flag_options=flag_options, is_hello=False)
    except TypeError:
        try:
            bootstrap.run(main_script, "", [], flag_options)  # type: ignore[arg-type]
        except TypeError:
            bootstrap.run(main_script, [], flag_options)  # type: ignore[arg-type]


# =========================================================
# Model download & loading
# =========================================================

def _ensure_model_file() -> Path:
    """
    Ensures the model file exists locally. If it does not, tries to download it.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return MODEL_PATH

    if not MODEL_URL:
        raise RuntimeError(
            "Model file was not found locally and MODEL_URL is not set. "
            "Either upload the model into a persistent path and set MODEL_PATH, "
            "or set MODEL_URL to a downloadable link."
        )

    if gdown is None:
        raise RuntimeError(
            "gdown is not installed, but MODEL_URL requires download support. "
            "Add `gdown` to requirements.txt or provide a direct HTTP URL."
        )

    tmp_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".partial")

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with st.spinner(f"Downloading model (attempt {attempt}/{DOWNLOAD_RETRIES})..."):
                gdown.download(MODEL_URL, str(tmp_path), quiet=False, fuzzy=True)

            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("Downloaded file is empty.")

            # Validate that it looks like an HDF5 file
            with h5py.File(str(tmp_path), "r"):
                pass

            tmp_path.replace(MODEL_PATH)
            return MODEL_PATH

        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

            if attempt >= DOWNLOAD_RETRIES:
                raise RuntimeError(f"Failed to download model: {e}") from e

            time.sleep(2 * attempt)

    return MODEL_PATH  # pragma: no cover


def _load_state_dict_from_h5(h5_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """
    Reads a PyTorch-like state_dict stored in an HDF5 file.
    Returns: (state_dict, meta)
    """
    state_dict: Dict[str, torch.Tensor] = {}
    meta: Dict[str, str] = {}

    with h5py.File(str(h5_path), "r") as f:
        if "weights" not in f:
            raise ValueError("HDF5 file does not contain the expected 'weights' group.")

        weights_group = f["weights"]

        for k in f.attrs.keys():
            meta[str(k)] = str(f.attrs[k])

        for key in weights_group.keys():
            arr = np.array(weights_group[key])
            tensor = torch.from_numpy(arr)
            state_dict[key] = tensor

    return state_dict, meta


# Streamlit cache compatibility
_CACHE_RESOURCE = getattr(st, "cache_resource", None)
if _CACHE_RESOURCE is None:
    # Older Streamlit fallback
    def _cache_resource_fallback(**_kwargs):
        return st.cache(allow_output_mutation=True)
    _CACHE_RESOURCE = _cache_resource_fallback


@_CACHE_RESOURCE(show_spinner=False)
def load_model_and_meta() -> Tuple[torch.nn.Module, Dict[str, str]]:
    """
    Loads and caches the model for the Streamlit process lifetime.
    """
    h5_path = _ensure_model_file()
    state_dict, meta = _load_state_dict_from_h5(h5_path)

    model = timm.create_model(TIMM_MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])

    if missing:
        meta["missing_keys"] = str(missing[:20]) + (" ..." if len(missing) > 20 else "")
    if unexpected:
        meta["unexpected_keys"] = str(unexpected[:20]) + (" ..." if len(unexpected) > 20 else "")

    model.eval()
    return model, meta


def preprocess_image(img: Image.Image, image_size: int = 224) -> torch.Tensor:
    img = img.convert("RGB").resize((image_size, image_size))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    x = torch.from_numpy(arr).unsqueeze(0)
    return x


@torch.inference_mode()
def predict(img: Image.Image) -> Tuple[int, np.ndarray]:
    model, _ = load_model_and_meta()
    x = preprocess_image(img)
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


# =========================================================
# Streamlit UI
# =========================================================

def render_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

    st.title(APP_TITLE)
    st.caption("Upload a cervical cytology image and receive a predicted class with confidence scores.")
    st.info(DISCLAIMER)

    with st.sidebar:
        st.header("System")
        st.subheader("Model status")
        st.write(f"**TIMM model:** `{TIMM_MODEL_NAME}`")
        st.write(f"**Model path:** `{MODEL_PATH}`")
        st.write(f"**Model URL configured:** `{bool(MODEL_URL)}`")

        if st.button("Clear Streamlit cache"):
            try:
                st.cache_resource.clear()  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                st.cache_data.clear()  # type: ignore[attr-defined]
            except Exception:
                pass
            st.success("Cache cleared.")

        st.divider()
        st.subheader("Diagnostics")
        try:
            exists = MODEL_PATH.exists()
            size_mb = (MODEL_PATH.stat().st_size / (1024 * 1024)) if exists else 0.0
            st.write(f"**Model file exists:** {exists}")
            if exists:
                st.write(f"**Model file size:** {size_mb:.1f} MB")
        except Exception as e:
            st.write(f"Diagnostics error: {e}")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1) Upload an image")
        uploaded = st.file_uploader("Supported formats: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])
        if uploaded is None:
            st.stop()
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)

    with col_right:
        st.subheader("2) Prediction")
        with st.spinner("Loading model and running inference..."):
            try:
                pred_idx, probs = predict(img)
                pred_label = CLASS_NAMES[pred_idx]
            except Exception as e:
                st.error("Inference failed. Please check Railway logs for details.")
                st.exception(e)
                st.stop()

        st.success(f"Predicted class: **{pred_label}**")
        st.write(CLASS_DESCRIPTIONS.get(pred_label, ""))

        rows = [{"Class": name, "Probability": float(probs[i])} for i, name in enumerate(CLASS_NAMES)]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("Probabilities are derived from softmax and may be miscalibrated depending on the training setup.")

    st.divider()
    st.subheader("About")
    st.write(
        "This demo runs a vision model for cervical cytology image classification. "
        "If you use a Railway Volume (persistent storage), set `MODEL_DIR` or `MODEL_PATH` "
        "to a mounted path (e.g., `/data/models`) so the model remains on the server across restarts."
    )


# =========================================================
# Entrypoint
# =========================================================

# When executed as `python main.py` (e.g., on Railway), bootstrap Streamlit.
if __name__ == "__main__" and not _in_streamlit_runtime():
    _bootstrap_streamlit_server()
    raise SystemExit(0)

# When executed by Streamlit, render the UI.
if _in_streamlit_runtime():
    render_app()
