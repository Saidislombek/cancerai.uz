from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import tensorflow as tf  # для keras-ветки

from app.model_registry import LoadedModel


@dataclass
class InferenceResult:
    model_id: str
    predicted_label: str
    predicted_probability: float
    probabilities: Dict[str, float]
    latency_ms: float
    input_shape: Tuple[int, int, int]  # (H,W,C)


def _default_labels(n: int) -> List[str]:
    base = ["HSIL", "LSIL", "NILM", "SCC"]
    if n == 4:
        return base
    return [f"class_{i}" for i in range(n)]


def _pil_to_timm_tensor(img: Image.Image, h: int, w: int, mean, std, device: str) -> torch.Tensor:
    img = img.convert("RGB").resize((w, h))
    arr = np.asarray(img).astype("float32") / 255.0  # HWC
    arr = (arr - np.array(mean, dtype="float32")) / np.array(std, dtype="float32")
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    x = torch.from_numpy(arr).unsqueeze(0)  # NCHW
    return x.to(device)


def run_inference(model_id: str, loaded: LoadedModel, image_bytes: bytes) -> InferenceResult:
    t0 = time.time()
    img = Image.open(io.BytesIO(image_bytes))

    # ===== timm_h5 branch (ваш текущий формат .h5) =====
    if loaded.kind == "timm_h5":
        h, w = loaded.input_hw
        device = loaded.device or ("cuda" if torch.cuda.is_available() else "cpu")

        x = _pil_to_timm_tensor(img, h, w, loaded.mean, loaded.std, device)

        with torch.no_grad():
            logits = loaded.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype("float64")

        labels = loaded.labels
        if len(labels) != int(probs.shape[0]):
            labels = _default_labels(int(probs.shape[0]))

        best_i = int(np.argmax(probs))
        out_probs = {labels[i]: float(probs[i]) for i in range(len(labels))}
        latency_ms = (time.time() - t0) * 1000.0

        return InferenceResult(
            model_id=model_id,
            predicted_label=labels[best_i],
            predicted_probability=float(probs[best_i]),
            probabilities=out_probs,
            latency_ms=latency_ms,
            input_shape=(h, w, 3),
        )

    # ===== keras branch (на будущее, если будет полноценная Keras модель) =====
    model = loaded.model
    h, w = loaded.input_hw

    img2 = img.convert("RGB").resize((w, h))
    arr = np.asarray(img2).astype("float32") / 255.0
    x = np.expand_dims(arr, axis=0)

    y = model.predict(x, verbose=0)[0]
    y = np.asarray(y).astype("float64")

    # если это логиты — приведём к вероятностям
    if (y < 0).any() or (y > 1).any() or abs(float(y.sum()) - 1.0) > 1e-2:
        y = tf.nn.softmax(y).numpy().astype("float64")

    labels = loaded.labels if loaded.labels else _default_labels(int(y.shape[0]))
    if len(labels) != int(y.shape[0]):
        labels = _default_labels(int(y.shape[0]))

    best_i = int(np.argmax(y))
    out_probs = {labels[i]: float(y[i]) for i in range(len(labels))}
    latency_ms = (time.time() - t0) * 1000.0

    return InferenceResult(
        model_id=model_id,
        predicted_label=labels[best_i],
        predicted_probability=float(y[best_i]),
        probabilities=out_probs,
        latency_ms=latency_ms,
        input_shape=(h, w, 3),
    )
