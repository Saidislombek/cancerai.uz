from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import tensorflow as tf

import torch
import timm
from timm.data import resolve_data_config


DEFAULT_LABELS = ["HSIL", "LSIL", "NILM", "SCC"]


def _as_str(x) -> str:
    # h5py attrs/datasets могут быть bytes / numpy scalar
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
    return str(x)


def _labels_for(model_id: str) -> List[str]:
    # LABELS_CC_VIT_STS_TESTED="HSIL,LSIL,NILM,SCC"
    raw = os.getenv(f"LABELS_{model_id.upper()}") or os.getenv("LABELS")
    if raw:
        labels = [s.strip() for s in raw.split(",") if s.strip()]
        if labels:
            return labels
    return DEFAULT_LABELS


def _timm_name_for(model_id: str) -> Optional[str]:
    # TIMM_MODEL_NAME_CC_ENV2_S_TESTED="tf_efficientnetv2_s" (пример)
    return os.getenv(f"TIMM_MODEL_NAME_{model_id.upper()}")


def _is_keras_h5(h5: h5py.File) -> bool:
    return "model_config" in h5.attrs or "keras_version" in h5.attrs


def _is_probably_keras_weights_only(h5: h5py.File) -> bool:
    # save_weights() часто даёт model_weights без model_config
    return ("model_weights" in h5) and (not _is_keras_h5(h5))


def _is_timm_like_h5(h5: h5py.File) -> bool:
    # принимаем больше вариантов
    return ("model_state_dict" in h5) or ("state_dict" in h5)


def _find_state_dict_group(h5: h5py.File) -> Optional[h5py.Group]:
    for key in ("model_state_dict", "state_dict"):
        if key in h5 and isinstance(h5[key], h5py.Group):
            return h5[key]
    return None


def _group_to_state_dict(group: h5py.Group, prefix: str = "") -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for k, obj in group.items():
        if isinstance(obj, h5py.Dataset):
            name = (prefix + k).replace("/", ".")
            arr = obj[()]
            if isinstance(arr, np.ndarray):
                t = torch.from_numpy(arr)
            else:
                t = torch.tensor(arr)
            sd[name] = t
        elif isinstance(obj, h5py.Group):
            sd.update(_group_to_state_dict(obj, prefix + k + "/"))
    return sd


def _read_model_name_from_h5(h5: h5py.File) -> Optional[str]:
    # 1) info/model_name
    if "info" in h5 and isinstance(h5["info"], h5py.Group):
        info = h5["info"]
        if "model_name" in info and isinstance(info["model_name"], h5py.Dataset):
            return _as_str(info["model_name"][()])
    # 2) root attrs
    for k in ("model_name", "arch", "timm_model", "backbone"):
        if k in h5.attrs:
            return _as_str(h5.attrs[k])
    # 3) root datasets
    for k in ("model_name", "arch", "timm_model", "backbone"):
        if k in h5 and isinstance(h5[k], h5py.Dataset):
            return _as_str(h5[k][()])
    return None


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    filename: str


@dataclass
class LoadedModel:
    kind: str  # "keras" | "timm_h5"
    model: object
    labels: List[str]
    input_hw: Tuple[int, int]  # (H, W)
    mean: Optional[Tuple[float, float, float]] = None
    std: Optional[Tuple[float, float, float]] = None


def _load_keras_h5(path: Path, labels: List[str]) -> LoadedModel:
    model = tf.keras.models.load_model(path, compile=False)
    # вход берём из модели (обычно (None,H,W,3))
    ishape = model.input_shape
    h = int(ishape[1]) if ishape and len(ishape) >= 3 else 224
    w = int(ishape[2]) if ishape and len(ishape) >= 3 else 224
    return LoadedModel(kind="keras", model=model, labels=labels, input_hw=(h, w))


def _load_timm_h5(path: Path, model_id: str, labels: List[str]) -> LoadedModel:
    with h5py.File(path, "r") as h5:
        model_name = _read_model_name_from_h5(h5) or _timm_name_for(model_id)
        if not model_name:
            raise RuntimeError(
                f"timm .h5 без model_name: {path}. "
                f"Добавь Railway Variable: TIMM_MODEL_NAME_{model_id.upper()}=\"...\" "
                f"(timm create_model name)."
            )

        sd_group = _find_state_dict_group(h5)
        if sd_group is None:
            raise RuntimeError(f"Не найден state_dict group в timm .h5: {path}")

        state_dict = _group_to_state_dict(sd_group)

    # строим модель и грузим веса
    model = timm.create_model(model_name, pretrained=False, num_classes=len(labels))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"State_dict missing keys for {model_name}: {missing[:20]} ...")
    if unexpected:
        # не критично, но полезно видеть
        pass

    model.eval()

    cfg = resolve_data_config({}, model=model)
    # cfg["input_size"] = (3,H,W)
    input_hw = (int(cfg["input_size"][1]), int(cfg["input_size"][2]))
    mean = tuple(float(x) for x in cfg.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(x) for x in cfg.get("std", (0.229, 0.224, 0.225)))

    return LoadedModel(
        kind="timm_h5",
        model=model,
        labels=labels,
        input_hw=input_hw,
        mean=mean,
        std=std,
    )


class ModelRegistry:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._lock = threading.Lock()
        self._cache: Dict[str, LoadedModel] = {}

        self.specs = self._default_specs()

    def _default_specs(self) -> List[ModelSpec]:
        return [
            ModelSpec("cc_env2_s_tested", "EfficientNetV2-S (EfficientNetV2 Small)", "cc_env2_s_tested.h5"),
            ModelSpec("cc_mswt_tested", "Multi-scale Window Transformer MSWT", "cc_mswt_tested.h5"),
            ModelSpec("cc_rf_tested", "Random Forest", "cc_rf_tested.h5"),
            ModelSpec("cc_ifs_tested", "IFS-kNN", "cc_ifs_tested.h5"),
            ModelSpec("cc_vit_sts_tested", "ViT Vision Transformer (Swin Transformer Small)", "cc_vit_sts_tested.h5"),
        ]

    def list_specs(self) -> List[ModelSpec]:
        return self.specs

    def get(self, model_id: str) -> LoadedModel:
        with self._lock:
            if model_id in self._cache:
                return self._cache[model_id]

            spec = next((s for s in self.specs if s.model_id == model_id), None)
            if spec is None:
                raise KeyError(f"Unknown model_id: {model_id}")

            path = self.model_dir / spec.filename
            if not path.exists():
                raise FileNotFoundError(f"Missing model file: {path}")

            labels = _labels_for(model_id)

            with h5py.File(path, "r") as h5:
                if _is_keras_h5(h5):
                    loaded = _load_keras_h5(path, labels)
                elif _is_probably_keras_weights_only(h5):
                    raise RuntimeError(
                        f"{path} похоже на Keras weights-only (save_weights), без model_config. "
                        f"Пересохрани модель через model.save(...) или дай файл в формате timm_h5."
                    )
                elif _is_timm_like_h5(h5):
                    loaded = _load_timm_h5(path, model_id, labels)
                else:
                    raise RuntimeError(
                        f"Unrecognized .h5 format: {path}. "
                        f"Expected Keras H5 (model_config) or timm-like H5 (model_state_dict/state_dict)."
                    )

            self._cache[model_id] = loaded
            return loaded
