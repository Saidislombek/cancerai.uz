from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

# Для Keras-моделей (если когда-то будут .keras / полноценные .h5 Keras)
import tensorflow as tf

# Для вашего формата .h5 (timm + torch state_dict)
import torch
import timm
from timm.data import resolve_data_config


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    filename: str  # имя файла внутри MODEL_DIR


@dataclass
class LoadedModel:
    kind: str  # "keras" или "timm_h5"
    model: object
    labels: List[str]
    input_hw: Tuple[int, int]  # (H, W)
    mean: Optional[Tuple[float, float, float]] = None
    std: Optional[Tuple[float, float, float]] = None
    device: Optional[str] = None


# Оставляем одну модель под ваш текущий файл
DEFAULT_MODELS: List[ModelSpec] = [
    ModelSpec("cc_vit_sts", "CC-ViT-STS (tested)", "cc_vit_sts_tested.h5"),
]


def _to_str(x) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    return str(x)


class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: Dict[str, LoadedModel] = {}
        self._specs: List[ModelSpec] = DEFAULT_MODELS

        # 1) Если задан MODEL_DIR — используем его
        env_dir = os.getenv("MODEL_DIR", "").strip()
        if env_dir:
            self._model_dir = Path(env_dir).resolve()
        else:
            # 2) Railway Volume: /data (создаём /data/models если /data существует)
            if Path("/data").exists():
                candidate = Path("/data/models")
                candidate.mkdir(parents=True, exist_ok=True)
                self._model_dir = candidate.resolve()
            else:
                # 3) Локальная разработка: <project_root>/models
                self._model_dir = (Path(__file__).resolve().parent.parent / "models").resolve()

    def list_specs(self) -> List[ModelSpec]:
        return list(self._specs)

    def resolve_path(self, model_id: str) -> Path:
        spec = next((m for m in self._specs if m.model_id == model_id), None)
        if spec is None:
            raise KeyError(f"Unknown model_id: {model_id}")
        return (self._model_dir / spec.filename).resolve()

    def clear_cache(self) -> None:
        with self._lock:
            self._models.clear()

    def _detect_kind(self, path: Path) -> str:
        # Keras H5 обычно содержит model_config / keras_version в attrs
        with h5py.File(path, "r") as f:
            if ("model_config" in f.attrs) or ("keras_version" in f.attrs):
                return "keras"
            # Ваш формат из Colab: группы info + model_state_dict
            if ("info" in f.keys()) and ("model_state_dict" in f.keys()):
                return "timm_h5"

        raise ValueError(
            f"Unrecognized .h5 format: {path}. "
            f"Expected Keras H5 (with model_config) or timm_h5 (info + model_state_dict)."
        )

    def _load_timm_h5(self, path: Path) -> LoadedModel:
        with h5py.File(path, "r") as f:
            info = f["info"]
            model_name = _to_str(info.attrs["model_name"])
            classes_csv = _to_str(info.attrs["classes"])
            labels = [c.strip() for c in classes_csv.split(",") if c.strip()]
            if not labels:
                raise ValueError("No classes found in info.attrs['classes']")

            num_classes = len(labels)

            device = "cuda" if torch.cuda.is_available() else "cpu"

            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes,
            ).to(device)

            # state_dict из H5
            g = f["model_state_dict"]
            state_dict = {}
            for name in g.keys():
                arr = np.array(g[name])
                state_dict[name] = torch.from_numpy(arr)

            # В вашем Colab было strict=True, оставляем так же
            model.load_state_dict(state_dict, strict=True)
            model.eval()

        # конфиг препроцессинга
        data_cfg = resolve_data_config({}, model=model)
        input_size = data_cfg.get("input_size", (3, 224, 224))  # (C,H,W)
        mean = tuple(float(x) for x in data_cfg.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(float(x) for x in data_cfg.get("std", (0.229, 0.224, 0.225)))
        h, w = int(input_size[1]), int(input_size[2])

        return LoadedModel(
            kind="timm_h5",
            model=model,
            labels=labels,
            input_hw=(h, w),
            mean=mean,
            std=std,
            device=device,
        )

    def _load_keras(self, path: Path) -> LoadedModel:
        model = tf.keras.models.load_model(str(path), compile=False)

        # labels можно задать через CLASSES="HSIL,LSIL,NILM,SCC"
        classes_env = os.getenv("CLASSES", "").strip()
        labels = [x.strip() for x in classes_env.split(",") if x.strip()] if classes_env else []

        # input size
        try:
            ish = model.input_shape  # (None,H,W,C)
            h, w = int(ish[1]), int(ish[2])
        except Exception:
            h, w = 224, 224

        return LoadedModel(kind="keras", model=model, labels=labels, input_hw=(h, w))

    def get(self, model_id: str) -> LoadedModel:
        with self._lock:
            if model_id in self._models:
                return self._models[model_id]

            path = self.resolve_path(model_id)
            if not path.exists():
                raise FileNotFoundError(f"Model file is missing: {path}")

            kind = self._detect_kind(path)
            if kind == "keras":
                loaded = self._load_keras(path)
            else:
                loaded = self._load_timm_h5(path)

            self._models[model_id] = loaded
            return loaded
