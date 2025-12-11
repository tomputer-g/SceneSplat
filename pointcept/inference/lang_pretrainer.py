import copy
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from pointcept.datasets.transform import Compose, TRANSFORMS
from pointcept.datasets.utils import collate_fn
from pointcept.engines.hooks.misc import CheckpointLoader
from pointcept.models import build_model
from pointcept.utils.config import Config


class LangPretrainerInference:
    """Standalone inference pipeline for LangPretrainer."""

    def __init__(
        self,
        cfg: Any,
        checkpoint_path: str,
        device: Optional[str] = None,
    ) -> None:
        if isinstance(cfg, (str, os.PathLike)):
            self.cfg = Config.fromfile(cfg)
        elif isinstance(cfg, Config):
            self.cfg = cfg
        else:
            raise TypeError("cfg must be a path or Config instance")

        if not hasattr(self.cfg, "inference"):
            raise ValueError("Inference config must define an `inference` section.")
        self.inference_cfg = self.cfg.inference

        feat_keys_cfg = self.cfg.get("feat_keys", ())
        if isinstance(feat_keys_cfg, (list, tuple)):
            self.feat_keys = tuple(feat_keys_cfg)
        elif feat_keys_cfg:
            self.feat_keys = (feat_keys_cfg,)
        else:
            self.feat_keys = ()

        self.logger = logging.getLogger("LangPretrainerInference")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)
            self.logger.propagate = False
        self.logger.setLevel(logging.INFO)

        requested_device = device or self.inference_cfg.get("device", "cuda")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning(
                "CUDA requested but not available; falling back to CPU device."
            )
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        self.chunk_size = self.inference_cfg.get("chunk_size", None)
        self.default_scene_name = self.inference_cfg.get(
            "default_scene_name", "inference_sample"
        )
        self.return_numpy = bool(self.inference_cfg.get("return_numpy", True))

        self.transform = Compose(self.inference_cfg.get("transform", []))
        test_cfg = self.inference_cfg.get("test_cfg", {})
        self.test_voxelize = (
            TRANSFORMS.build(test_cfg["voxelize"])
            if test_cfg and test_cfg.get("voxelize")
            else None
        )
        self.test_crop = (
            TRANSFORMS.build(test_cfg["crop"])
            if test_cfg and test_cfg.get("crop")
            else None
        )
        post_transform_cfg = test_cfg.get("post_transform", []) if test_cfg else []
        self.post_transform = Compose(post_transform_cfg)
        aug_transform_cfg = test_cfg.get("aug_transform", [[]]) if test_cfg else [[]]
        self.aug_transform = [Compose(cfg_) for cfg_ in aug_transform_cfg]

        self.save_cfg = self.inference_cfg.get("save_features", {}) or {}
        self.output_dir = self.save_cfg.get("output_dir")
        backbone_cfg = self.save_cfg.get("backbone", {}) or {}
        self.save_backbone = bool(backbone_cfg.get("enabled", False))
        self.backbone_save_cfg = backbone_cfg

        self.model = build_model(self.cfg.model)
        self.model.eval()
        self.model.to(self.device)
        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        loader = CheckpointLoader(strict=False)
        trainer = SimpleNamespace(
            cfg=SimpleNamespace(weight=checkpoint_path, resume=False),
            model=self.model,
            logger=self.logger,
        )
        loader.trainer = trainer
        loader.before_train()

    def __call__(
        self,
        data: Dict[str, np.ndarray],
        *,
        scene_name: Optional[str] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        prepared = self._prepare_input_dict(data, scene_name)
        fragments = prepared.pop("fragment_list")
        scene = prepared.get("name", self.default_scene_name)

        if not fragments:
            raise RuntimeError("Inference requires at least one fragment to process.")

        num_points = (
            prepared["segment"].shape[0]
            if prepared.get("segment") is not None
            else fragments[0]["coord"].shape[0]
        )
        feature_acc = None
        feature_count = None

        for fragment in fragments:
            input_dict = collate_fn([fragment])
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.to(self.device, non_blocking=True)

            with torch.no_grad():
                forward_kwargs = {"return_backbone": True}
                if self.chunk_size:
                    forward_kwargs["chunk_size"] = self.chunk_size
                out_dict = self.model(input_dict, **forward_kwargs)

            feat_tensor = self._get_feature_tensor(out_dict)
            if feat_tensor is None:
                raise RuntimeError("Model did not return backbone features.")

            idx_part = input_dict["index"]
            offset_list = input_dict["offset"]

            start = 0
            for end in offset_list:
                slice_idx = idx_part[start:end]
                feat_slice = feat_tensor[start:end]
                if feature_acc is None:
                    feature_acc = torch.zeros(
                        (num_points, feat_slice.size(1)),
                        device=feat_slice.device,
                        dtype=feat_slice.dtype,
                    )
                    feature_count = torch.zeros(
                        num_points,
                        device=feat_slice.device,
                        dtype=feat_slice.dtype,
                    )
                feature_acc[slice_idx] += feat_slice
                feature_count[slice_idx] += 1
                start = end

        if feature_acc is None or feature_count is None:
            raise RuntimeError("Failed to accumulate any features.")

        valid = feature_count > 0
        if torch.any(valid):
            feature_acc[valid] = feature_acc[valid] / feature_count[valid].unsqueeze(1)

        inverse_map = prepared.get("inverse")
        if inverse_map is not None:
            inverse_tensor = torch.as_tensor(
                inverse_map, device=feature_acc.device, dtype=torch.long
            )
            feature_acc = feature_acc[inverse_tensor]

        feature_acc = F.normalize(feature_acc, p=2, dim=1)
        backbone_cpu = feature_acc.detach().cpu()

        outputs = {
            "name": scene,
            "backbone_features": backbone_cpu.numpy()
            if self.return_numpy
            else backbone_cpu,
            "metadata": {
                "origin_coord": prepared.get("origin_coord"),
                "origin_segment": prepared.get("origin_segment"),
                "origin_feat_mask": prepared.get("origin_feat_mask"),
                "inverse": inverse_map,
            },
        }

        if save and self.save_backbone:
            self._save_backbone(scene, backbone_cpu)

        return outputs

    def _get_feature_tensor(self, model_output: Dict[str, Any]) -> Optional[torch.Tensor]:
        tensor = model_output.get("backbone_feat")
        if tensor is not None:
            return tensor
        point_feat = model_output.get("point_feat")
        if point_feat is None:
            return None
        if isinstance(point_feat, dict):
            return point_feat.get("feat")
        if hasattr(point_feat, "feat"):
            return point_feat.feat
        if isinstance(point_feat, torch.Tensor):
            return point_feat
        return None

    def _prepare_input_dict(
        self, data: Dict[str, np.ndarray], scene_name: Optional[str]
    ) -> Dict[str, Any]:
        base_dict = self._format_numpy_inputs(data, scene_name)
        data_dict = self.transform(base_dict)

        result = dict(
            segment=data_dict.pop("segment", None),
            name=data_dict.pop("name", scene_name or self.default_scene_name),
        )
        if "coord" in data_dict:
            result["coord"] = data_dict["coord"]
        if "pc_coord" in data_dict:
            result["pc_coord"] = data_dict["pc_coord"]
        if "pc_segment" in data_dict:
            result["pc_segment"] = data_dict["pc_segment"]
        if "origin_coord" in data_dict:
            result["origin_coord"] = data_dict.pop("origin_coord")
        if "origin_feat_mask" in data_dict:
            result["origin_feat_mask"] = data_dict.pop("origin_feat_mask")
        if "origin_instance" in data_dict:
            result["origin_instance"] = data_dict.pop("origin_instance")
        if "origin_segment" in data_dict:
            result["origin_segment"] = data_dict.pop("origin_segment")
        if "inverse" in data_dict:
            result["inverse"] = data_dict.pop("inverse")

        fragments = []
        for aug in self.aug_transform:
            fragments.append(aug(copy.deepcopy(data_dict)))

        fragment_list = []
        for fragment in fragments:
            if self.test_voxelize is not None:
                data_slices = self.test_voxelize(fragment)
            else:
                fragment["index"] = np.arange(fragment["coord"].shape[0])
                data_slices = [fragment]
            for slice_data in data_slices:
                if self.test_crop is not None:
                    crop_list = self.test_crop(slice_data)
                else:
                    crop_list = [slice_data]
                fragment_list.extend(crop_list)

        fragment_list = [self.post_transform(frag) for frag in fragment_list]
        result["fragment_list"] = fragment_list
        return result

    def _format_numpy_inputs(
        self, data: Dict[str, np.ndarray], scene_name: Optional[str]
    ) -> Dict[str, Any]:
        if self.feat_keys:
            missing = [key for key in self.feat_keys if key not in data]
            if missing:
                raise KeyError(
                    "Missing required features: " + ", ".join(sorted(set(missing)))
                )

        formatted: Dict[str, Any] = {}

        num_points = None
        if "coord" in data:
            coord = np.asarray(data["coord"], dtype=np.float32)
            formatted["coord"] = coord
            num_points = coord.shape[0]
        else:
            for key in self.feat_keys:
                if key in data:
                    reference = np.asarray(data[key])
                    num_points = reference.shape[0]
                    break
            if num_points is None:
                raise KeyError(
                    "Unable to determine point count; provide 'coord' or a feature listed in feat_keys."
                )

        def _ensure_rows(name: str, array: np.ndarray) -> np.ndarray:
            if array.shape[0] != num_points:
                raise ValueError(f"{name} shape mismatch: expected {num_points} rows.")
            return array

        def _assign_float(name: str):
            array = np.asarray(data[name], dtype=np.float32)
            formatted[name] = _ensure_rows(name, array)

        for key in ["color", "quat", "scale"]:
            if key in data:
                _assign_float(key)

        if "opacity" in data:
            opacity = np.asarray(data["opacity"], dtype=np.float32)
            if opacity.ndim == 1:
                opacity = opacity.reshape(-1, 1)
            formatted["opacity"] = _ensure_rows("opacity", opacity)

        if "normal" in data:
            _assign_float("normal")

        if "segment" in data:
            segment = np.asarray(data["segment"])
            if segment.ndim == 2:
                segment = segment[:, 0]
            segment = segment.reshape(-1).astype(np.int32)
            formatted["segment"] = _ensure_rows("segment", segment)
        else:
            formatted["segment"] = np.full((num_points,), -1, dtype=np.int32)

        if "instance" in data:
            instance = np.asarray(data["instance"])
            if instance.ndim == 2:
                instance = instance[:, 0]
            instance = instance.reshape(-1).astype(np.int32)
            formatted["instance"] = _ensure_rows("instance", instance)
        else:
            formatted["instance"] = np.full((num_points,), -1, dtype=np.int32)

        if "valid_feat_mask" in data:
            mask = np.asarray(data["valid_feat_mask"]).astype(bool)
            formatted["valid_feat_mask"] = _ensure_rows("valid_feat_mask", mask)

        formatted["name"] = scene_name or data.get("name", self.default_scene_name)
        return formatted

    def _save_backbone(self, scene: str, tensor: torch.Tensor) -> None:
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        file_name = self.backbone_save_cfg.get("file_name", "feat.pt")
        path = self._resolve_output_path(scene, file_name)
        self._dump_tensor(tensor.half(), path)

    def _resolve_output_path(self, scene: str, file_name: str) -> str:
        if "{" in file_name:
            file_name = file_name.format(scene=scene)
        else:
            file_name = f"{scene}_{file_name}"
        return os.path.join(self.output_dir, file_name)

    def _dump_tensor(self, tensor: torch.Tensor, path: str) -> None:
        if path.endswith(".npy"):
            np.save(path, tensor.numpy())
        else:
            torch.save(tensor, path)
        self.logger.info(f"Saved features with shape {tensor.shape} to {path}")


__all__ = ["LangPretrainerInference"]
