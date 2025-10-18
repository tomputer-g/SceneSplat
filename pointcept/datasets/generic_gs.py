import os
import numpy as np

from pointcept.utils.cache import shared_dict

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class GenericGSDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment",
        "quat",
        "scale",
        "opacity",
        "instance",
    ]
    EVAL_PC_ASSETS = ["pc_coord", "pc_segment", "pc_instance"]

    def __init__(
        self,
        multilabel=False,
        is_train=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multilabel = multilabel
        self.is_train = is_train

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if self.is_train:
                if asset[:-4] not in self.VALID_ASSETS:
                    continue
            else:
                if (
                    asset[:-4] not in self.VALID_ASSETS
                    and asset[:-4] not in self.EVAL_PC_ASSETS
                ):
                    continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "pc_coord" in data_dict.keys():
            data_dict["pc_coord"] = data_dict["pc_coord"].astype(np.float32)

        if "pc_segment" in data_dict.keys():
            data_dict["pc_segment"] = data_dict["pc_segment"].astype(np.int32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "opacity" in data_dict.keys():
            data_dict["opacity"] = data_dict["opacity"].astype(np.float32).clip(0.001)
            data_dict["opacity"] = data_dict["opacity"].reshape(-1, 1)

        if "quat" in data_dict.keys():
            data_dict["quat"] = data_dict["quat"].astype(np.float32)

        if "scale" in data_dict.keys():
            data_dict["scale"] = (
                data_dict["scale"].astype(np.float32).clip(1e-4, 1.0)
            )  # clip scale

        if "segment" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment").reshape([-1]).astype(np.int32)
            )
        
        if "instance" in data_dict.keys():
            data_dict["instance"] = (
                data_dict.pop("instance").reshape([-1]).astype(np.int32)
            )

        return data_dict
