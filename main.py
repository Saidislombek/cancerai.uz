"""
Cervical Cytology AI — Streamlit application entrypoint.

This file is designed to work in two modes:

1) Streamlit mode (recommended locally):
   streamlit run main.py

2) "Plain Python" mode (useful for Railway/other PaaS):
   python main.py

In plain Python mode, the script will automatically re-launch itself via:
   python -m streamlit run main.py ...

This avoids having to configure "streamlit run" as the platform start command,
while still running the Streamlit server correctly.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn.functional as F
import timm
import h5py
import gdown
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("cancerai")


# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
APP_TITLE = "Cervical Cytology AI"
APP_SUBTITLE = "Cervical cytology phenotype prediction from microscopy images"

# Recommended label order for your project (kept as a fallback).
DEFAULT_CLASS_ORDER = ["HSIL", "LSIL", "NILM", "SCC"]

# Use a persistent directory when possible.
# On Railway, attach a Volume and mount it at /data (or keep the default).
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/data/models")).resolve()
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "cc_vit_sts.h5")
MODEL_PATH = (MODEL_DIR / MODEL_FILENAME).resolve()

# Provide your Google Drive (or any HTTPS) download URL via env var.
# Example (Google Drive):
#   https://drive.google.com/uc?id=<FILE_ID>
MODEL_URL = os.getenv("MODEL_URL", "").strip()

# Streamlit server settings for PaaS environments.
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", "8501"))


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _in_streamlit_runtime() -> bool:
    """Return True if the script is currently executed by Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _relaunch_with_streamlit() -> None:
    """
    Relaunch this file using Streamlit when the platform starts it via:
        python main.py
    """
    args = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.address",
        SERVER_HOST,
        "--server.port",
        str(SERVER_PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    log.info("Re-launching via Streamlit: %s", " ".join(args))
    os.execv(sys.executable, args)


def ensure_model_file(model_path: Path, model_url: str) -> Path:
    """
    Ensure the model exists on disk.

    - If the file exists, do nothing.
    - If not, download it once (recommended to store in a persistent volume).
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    if not model_url:
        raise RuntimeError(
            "Model file is missing and MODEL_URL is not set. "
            "Set MODEL_URL in Railway -> Variables (or provide the model file in the image/volume)."
        )

    log.info("Model not found. Downloading to: %s", model_path)
    # gdown supports both file IDs and direct links (best: https://drive.google.com/uc?id=<id>)
    out = gdown.download(url=model_url, output=str(model_path), quiet=False, fuzzy=True)
    if not out or not model_path.exists():
        raise RuntimeError("Model download failed. Check MODEL_URL and file permissions (must be accessible).")

    log.info("Model downloaded successfully: %s (%.1f MB)", model_path, model_path.stat().st_size / (1024 * 1024))
    return model_path


def _decode_classes_attr(value) -> List[str]:
    """
    Decode HDF5 attribute "classes" into a list of strings.
    Handles bytes, numpy arrays, and JSON-encoded lists.
    """
    if value is None:
        return []
    try:
        if isinstance(value, (bytes, bytearray)):
            s = value.decode("utf-8", errors="ignore").strip()
            # Try JSON first
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
            # Fallback: comma-separated
            return [x.strip() for x in s.split(",") if x.strip()]

        if isinstance(value, (list, tuple, np.ndarray)):
            out: List[str] = []
            for x in value:
                if isinstance(x, (bytes, bytearray)):
                    out.append(x.decode("utf-8", errors="ignore"))
                else:
                    out.append(str(x))
            return out

        return [str(value)]
    except Exception:
        return []


def load_model_and_metadata(model_path: Path) -> Tuple[torch.nn.Module, List[str], Dict]:
    """
    Load a timm model and its state_dict from an HDF5 (.h5) file.
    The file is expected to contain:
      - attrs["classes"] (optional)
      - attrs["model_config"] (optional JSON)
      - group "model_state_dict" with datasets for each tensor
    """
    with h5py.File(str(model_path), "r") as f:
        classes_raw = f.attrs.get("classes", None)
        class_names = _decode_classes_attr(classes_raw) or DEFAULT_CLASS_ORDER

        cfg_raw = f.attrs.get("model_config", None)
        model_cfg: Dict = {}
        if cfg_raw is not None:
            try:
                if isinstance(cfg_raw, (bytes, bytearray)):
                    model_cfg = json.loads(cfg_raw.decode("utf-8", errors="ignore"))
                elif isinstance(cfg_raw, str):
                    model_cfg = json.loads(cfg_raw)
            except Exception:
                model_cfg = {}

        model_name = model_cfg.get("model_name", "swin_small_patch4_window7_224")
        num_classes = int(model_cfg.get("num_classes", len(class_names)))
        pretrained = bool(model_cfg.get("pretrained", False))

        log.info("Loading model: name=%s num_classes=%s pretrained=%s", model_name, num_classes, pretrained)
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        # Load state_dict
        if "model_state_dict" not in f:
            raise RuntimeError('Invalid model file: group "model_state_dict" not found.')

        state_group = f["model_state_dict"]
        state_dict = {}
        for key in state_group.keys():
            arr = state_group[key][()]
            tensor = torch.tensor(arr)
            state_dict[key] = tensor

        model.load_state_dict(state_dict, strict=True)
        model.eval()

    return model, class_names, model_cfg


def build_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )


@st.cache_resource(show_spinner=False)
def get_inference_bundle() -> Tuple[torch.nn.Module, List[str], transforms.Compose]:
    """
    Cache the model and preprocessing pipeline across reruns.
    """
    model_file = ensure_model_file(MODEL_PATH, MODEL_URL)
    model, class_names, cfg = load_model_and_metadata(model_file)

    img_size = int(cfg.get("img_size", 224)) if isinstance(cfg, dict) else 224
    tfm = build_transform(img_size=img_size)
    return model, class_names, tfm


def predict(image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
    """
    Run inference and return:
      - predicted label
      - confidence (0..1)
      - probability per class
    """
    model, class_names, tfm = get_inference_bundle()

    x = tfm(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy().reshape(-1)

    # Align lengths safely
    n = min(len(class_names), len(probs))
    class_names = class_names[:n]
    probs = probs[:n]

    best_idx = int(np.argmax(probs))
    best_label = class_names[best_idx]
    best_conf = float(probs[best_idx])

    prob_map = {cls: float(p) for cls, p in zip(class_names, probs)}
    return best_label, best_conf, prob_map


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
PHENOTYPE_INFO = {
    "NILM": (
        "Negative for intraepithelial lesion or malignancy (NILM). "
        "This suggests no cytological evidence of a precancerous lesion or malignancy."
    ),
    "LSIL": (
        "Low-grade squamous intraepithelial lesion (LSIL). "
        "Often associated with transient HPV infection; follow local clinical guidelines."
    ),
    "HSIL": (
        "High-grade squamous intraepithelial lesion (HSIL). "
        "Indicates a higher risk of significant precancerous changes; clinical follow-up is essential."
    ),
    "SCC": (
        "Squamous cell carcinoma (SCC). "
        "A malignant category; urgent clinical evaluation is required."
    ),
}


def render_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="centered")

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    st.info(
        "Disclaimer: This tool is for research/educational use only and does not provide medical advice. "
        "Always consult a qualified clinician for diagnosis and treatment decisions."
    )

    with st.expander("How it works", expanded=False):
        st.write(
            "1) Upload a cervical cytology microscopy image.\n"
            "2) The model performs a single-image inference.\n"
            "3) You receive a predicted phenotype and class probabilities."
        )
        st.write(
            "Operational note: The model file is downloaded once (if missing) and then cached in memory for faster reruns."
        )

    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if not uploaded:
        st.stop()

    try:
        image = Image.open(uploaded)
    except Exception as e:
        st.error(f"Could not read the uploaded image: {e}")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        run = st.button("Run inference", type="primary", use_container_width=True)
    with col2:
        st.write("")  # spacing

    if not run:
        st.stop()

    with st.spinner("Running inference..."):
        try:
            label, conf, prob_map = predict(image)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    st.success(f"Prediction: **{label}**  |  Confidence: **{conf:.2%}**")

    # Explanation
    st.subheader("Interpretation")
    st.write(PHENOTYPE_INFO.get(label, "No description is available for this label."))

    # Probabilities table
    st.subheader("Class probabilities")
    prob_items = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
    st.table(
        {
            "Class": [k for k, _ in prob_items],
            "Probability": [f"{v:.4f}" for _, v in prob_items],
        }
    )

    st.divider()
    st.caption("© CancerAI — Streamlit on Railway")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # If the platform runs `python main.py`, we re-launch the Streamlit server.
    # If Streamlit is already running (script executed by Streamlit), render the app normally.
    if not _in_streamlit_runtime():
        _relaunch_with_streamlit()
    else:
        render_app()
else:
    # When imported by Streamlit (rare), render the app.
    render_app()
