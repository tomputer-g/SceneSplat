"""
example commands for inference:
PYTHONPATH=. python tools/lang_inference.py \
    --config configs/inference/lang-pretrain-pt-v3m1-3dgs.py \
    --checkpoint ckpts/model_best.pth \
    --input-root /path/to/a/preprocessed/3dgs/npy/folder \
    --output-dir /output/path
"""

import argparse
import json
import os

import numpy as np

from pointcept.inference import LangPretrainerInference
from pointcept.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone inference for LangPretrainer."
    )
    parser.add_argument("--config", required=True, help="Path to inference config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file to load.")
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory that stores processed Gaussian .npy files.",
    )
    parser.add_argument("--scene-name", default=None, help="Optional scene name.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory defined in config.save_features.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving even if config enables it.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device string such as cpu or cuda:0.",
    )
    parser.add_argument(
        "--dump-json",
        default=None,
        help="Optional path to save a JSON summary of produced features.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    feat_keys = cfg.get("feat_keys", None)
    if not feat_keys:
        raise KeyError("Inference config must define `feat_keys`.")

    if not os.path.isdir(args.input_root):
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    data_dict = {}
    for file_name in os.listdir(args.input_root):
        if not file_name.endswith(".npy"):
            continue
        key = os.path.splitext(file_name)[0]
        file_path = os.path.join(args.input_root, file_name)
        data_dict[key] = np.load(file_path)

    missing = [k for k in feat_keys if k not in data_dict]
    if missing:
        raise FileNotFoundError(
            "Missing required feature files: " + ", ".join(f"{m}.npy" for m in missing)
        )

    scene_name = args.scene_name or os.path.basename(os.path.normpath(args.input_root))

    inferencer = LangPretrainerInference(
        cfg,
        args.checkpoint,
        device=args.device,
    )
    if args.output_dir is not None:
        inferencer.output_dir = args.output_dir

    outputs = inferencer(
        data_dict,
        scene_name=scene_name,
        save=not args.no_save,
    )

    backbone = outputs["backbone_features"]
    summary = {
        "name": outputs["name"],
        "backbone_features_shape": (
            list(backbone.shape) if backbone is not None else None
        ),
        "metadata_keys": list(outputs["metadata"].keys()),
    }

    if args.dump_json:
        with open(args.dump_json, "w") as f:
            json.dump(summary, f, indent=2)
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
