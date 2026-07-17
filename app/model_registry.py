from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

# Keras-модели (если будут .keras / полноценные .h5 Keras)
import tensorflow as tf

# Ваш формат .h5 (timm + torch state_dict)
import torch
import timm
from timm.data import resolve_data_config


@dataclass(frozen=True)
class ModelSpec:
    model_id: str          # значение, которое пойдёт в запрос (value в <option>)
    display_name: str      # то, что увидите в dropdown на фронтенде
    filename: str          # имя файла в папке MODEL_DIR (/data/models)


@dataclass
class LoadedModel:
    kind: str  # "keras" или "timm_h5"
    model: object
    labels: List[str]
    input_hw: Tuple[int, int]  # (H, W)
    mean: Optional[Tuple[float, float, float]] = None
    std: Optional[Tuple[float, float, float]] = None
    device: Optional[str] = None


# 5 моделей, как вы хотите видеть на фронтенде
DEFAULT_MODELS: List[ModelSpec] = [
    ModelSpec(
        model_id="cc_mswt_tested",
        display_name="Многомасштабный оконный трансформер (MSWT)",
        filename="cc_mswt_tested.h5",
    ),
    ModelSpec(
        model_id="cc_env2_s_tested",
        display_name="EfficientNetV2-S",
        filename="cc_env2_s_tested.h5",
    ),
    ModelSpec(
        model_id="cc_rf_tested",
        display_name="Случайный лес (Random Forest)",
        filename="cc_rf_tested.h5",
    ),
    ModelSpec(
        model_id="cc_ifs_tested",
        display_name="IFS-kNN",
        filename="cc_ifs_tested.h5",
    ),
    ModelSpec(
        model_id="cc_vit_sts_tested",
        display_name="Визуальный трансформер ViT (Swin Transformer Small)",
        filename="cc_vit_sts_tested.h5",
    ),
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

        # 1) Если задан MODEL_DIR — используем его (Railway: /data/models)
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
            raise KeyError(f"Неизвестный идентификатор модели: {model_id}")
        return (self._model_dir / spec.filename).resolve()

    def clear_cache(self) -> None:
        with self._lock:
            self._models.clear()

    def _detect_kind(self, path: Path) -> str:
        with h5py.File(path, "r") as f:
            # Keras H5 обычно содержит model_config / keras_version в attrs
            if ("model_config" in f.attrs) or ("keras_version" in f.attrs):
                return "keras"
            # Ваш формат из Colab: группы info + model_state_dict
            if ("info" in f.keys()) and ("model_state_dict" in f.keys()):
                return "timm_h5"

        raise ValueError(
            f"Неизвестный формат .h5: {path}. "
            f"Ожидается Keras H5 (с model_config) или timm_h5 (info + model_state_dict)."
        )

    def _load_timm_h5(self, path: Path) -> LoadedModel:
        with h5py.File(path, "r") as f:
            info = f["info"]
            model_name = _to_str(info.attrs["model_name"])
            classes_csv = _to_str(info.attrs["classes"])
            labels = [c.strip() for c in classes_csv.split(",") if c.strip()]
            if not labels:
                raise ValueError("В атрибуте info.attrs['classes'] не найдены классы")

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

            model.load_state_dict(state_dict, strict=True)
            model.eval()

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
                raise FileNotFoundError(f"Файл модели не найден: {path}")

            kind = self._detect_kind(path)
            if kind == "keras":
                loaded = self._load_keras(path)
            else:
                loaded = self._load_timm_h5(path)

            self._models[model_id] = loaded
            return loaded
