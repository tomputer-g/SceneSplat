import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
import copy


from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
    neighbor_voting,
    clustering_voting,
)


TESTERS = Registry("testers")


class TesterBase:
    def __init__(
        self, cfg, model=None, test_loader=None, verbose=True, index=None
    ) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model(index=index)
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader(index=index)
        else:
            self.test_loader = test_loader

    def build_model(self, index=None):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self, index=None):
        if index is not None:
            test_dataset = build_dataset(self.cfg.data.test[index])
        else:
            test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=0,  # temp, self.cfg.batch_size_test_per_gpu
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class ZeroShotSemSegTester(TesterBase):
    def __init__(
        self,
        cfg,
        model=None,
        test_loader=None,
        verbose=True,
        class_names=None,
        text_embeddings=None,
        excluded_classes=None,
        enable_voting=False,
        vote_k=25,
        confidence_threshold=0.1,
        ignore_index=-1,
        save_feat=False,
        skip_eval=False,
        pred_label_mapping=None,
        **kwargs,
    ):
        super().__init__(cfg, model, test_loader, verbose, **kwargs)
        if "index" in kwargs:
            # multi-dataset testing
            cfg = copy.deepcopy(cfg)
            cfg["test"] = cfg["test"][kwargs["index"]]
            cfg.data.test = cfg.data.test[kwargs["index"]]
        self.cfg = cfg
        self.enable_voting = cfg["test"].get("enable_voting", enable_voting)
        self.vote_k = cfg["test"].get("vote_k", vote_k)
        self.confidence_threshold = cfg["test"].get(
            "confidence_threshold", confidence_threshold
        )
        self.save_feat = cfg["test"].get("save_feat", save_feat)
        self.skip_eval = cfg["test"].get("skip_eval", skip_eval)
        self.pred_label_mapping = cfg["test"].get(
            "pred_label_mapping", pred_label_mapping
        )
        self.ignore_index = ignore_index

        class_names = cfg["test"].get("class_names", class_names)
        text_embeddings = cfg["test"].get("text_embeddings", text_embeddings)
        excluded_classes = cfg["test"].get("excluded_classes", excluded_classes)

        # Load class names and text embeddings
        if class_names:
            with open(class_names, "r") as f:
                self.class_names = [line.strip() for line in f if line.strip()]
        else:
            self.class_names = []
        if text_embeddings:
            self.text_embeddings = torch.load(text_embeddings, weights_only=True).cuda()
            self.text_embeddings = F.normalize(self.text_embeddings, p=2, dim=1)
        else:
            self.text_embeddings = None

        # Handle excluded classes
        self.excluded_indices, self.keep_indices = [], []
        if excluded_classes:
            self.excluded_indices = [
                i
                for i, name in enumerate(self.class_names)
                if name in (excluded_classes or [])
            ]
            self.keep_indices = [
                i
                for i in range(len(self.class_names))
                if i not in self.excluded_indices
            ]
        self.num_keep_classes = len(self.keep_indices)
        self.num_classes = len(self.class_names)
        if self.pred_label_mapping is None and not self.skip_eval:
            assert self.num_classes == self.text_embeddings.size(0), (
                "Mismatch in class names and text embeddings"
            )

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(
            ">>>>>>>>>>>>>> ZeroShotSemSegTester Start Evaluation >>>>>>>>>>>>>"
        )
        logger.info(
            f"Testing on {self.cfg.data.test.split} split of {self.cfg.data.test.type}"
        )
        if not self.skip_eval:
            logger.info(
                f"ZeroShotSemSegTester loaded text embeddings with shape {self.text_embeddings.shape}"
            )
        else:
            logger.info("ZeroShotSemSegTester skipping evaluation...")
        if self.enable_voting:
            logger.info("Neighbor voting enabled with k={}".format(self.vote_k))
        if self.save_feat:
            logger.info("Saving inference feature enabled")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(
            self.cfg.save_path, f"result_{self.cfg.data.test.type}"
        )
        make_dirs(save_path)

        # ================ Preserved Submission Handling ================
        # Create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
            or self.cfg.data.test.type == "ScanNetPPDataset"
            or "ScanNetPP" in self.cfg.data.test.type
            or "ScanNet" in self.cfg.data.test.type
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        if self.save_feat:
            make_dirs(os.path.join(save_path, "feat"))
        comm.synchronize()
        # ================ End Submission Handling ================
        record = {}
        # Fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment", None)
            data_name = data_dict.pop("name", "default")
            pred_save_path = os.path.join(save_path, f"{data_name}_pred.npy")
            feat_save_path = (
                os.path.join(save_path, "feat", f"{data_name}_feat.pth")
                if self.save_feat
                else None
            )

            if (
                os.path.isfile(pred_save_path)
                and not self.save_feat
                and "pc_coord" not in data_dict
            ):
                # note, pred later is saved for pc_coord, need inference again to evaluate metrics
                logger.info(f"{data_name}: loaded existing pred")
                pred = np.load(pred_save_path)
                if "pc_segment" in data_dict and "pc_coord" in data_dict:
                    segment = data_dict["pc_segment"]
                elif "origin_segment" in data_dict:
                    segment = data_dict["origin_segment"]
                if (
                    self.cfg.data.test.type == "ScanNetPPDataset"
                    or "ScanNetPP" in self.cfg.data.test.type
                ):
                    pred = pred[:, 0]  # we save top-3 classes for ScanNetPP
            else:
                num_points = (
                    segment.size if segment is not None else data_dict["coord"].shape[0]
                )
                num_classes = (
                    self.text_embeddings.size(0)
                    if self.text_embeddings is not None
                    else 10
                )  # placeholder if None
                ignore_index = self.ignore_index

                # Create a buffer to accumulate probabilities (or logits)
                pred = torch.zeros((num_points, num_classes), device="cuda")
                pred_coord = torch.zeros((num_points, 3), device="cuda")

                if self.save_feat:
                    feat_dim = (
                        self.text_embeddings.shape[1]
                        if self.text_embeddings is not None
                        else 768
                    )
                    point_features = torch.zeros((num_points, feat_dim), device="cuda")
                    feature_counts = torch.zeros(num_points, device="cuda")

                # ---------------------------------------------------------------------
                # Accumulate probabilities from each fragment
                # ---------------------------------------------------------------------
                for i, frag_dict in enumerate(fragment_list):
                    # collate => partial data
                    input_dict = collate_fn([frag_dict])
                    # move to GPU
                    for key in input_dict:
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)

                    idx_part = input_dict["index"]
                    offset_list = input_dict["offset"]

                    # Forward pass
                    with torch.no_grad():
                        out_dict = self.model(input_dict, chunk_size=600000)
                        # e.g., point feature [M, feat_dim]
                        pred_part_feat = out_dict["point_feat"][
                            "feat"
                        ]  # shape [M, feat_dim]
                        if not self.skip_eval:
                            logits = torch.mm(pred_part_feat, self.text_embeddings.t())
                            pred_part_prob = torch.sigmoid(logits)  # [M, num_classes]

                    # Accumulate into the large buffer
                    bs = 0
                    for be in offset_list:
                        if not self.skip_eval:
                            # sum up probabilities
                            pred[idx_part[bs:be], :] += pred_part_prob[bs:be]
                            # track coords if needed
                            pred_coord[idx_part[bs:be]] = input_dict["coord"][bs:be]
                        if self.save_feat:
                            point_features[idx_part[bs:be], :] += pred_part_feat[bs:be]
                            feature_counts[idx_part[bs:be]] += 1
                        bs = be

                    logger.info(
                        f"Test: {idx + 1}/{len(self.test_loader)}-{data_name}, "
                        f"Fragment batch: {i + 1}/{len(fragment_list)}"
                    )

                if self.save_feat:
                    pred_mask = feature_counts > 0
                    # points appear multiple times together in fragments
                    point_features[pred_mask] /= feature_counts[pred_mask].unsqueeze(1)
                    if "origin_coord" in data_dict and "inverse" in data_dict:
                        point_features_cpu = point_features.cpu()
                        point_features_cpu = F.normalize(point_features_cpu, p=2, dim=1)
                        final_features = point_features_cpu[data_dict["inverse"]]
                    else:
                        final_features = F.normalize(point_features, p=2, dim=1)
                    torch.save(final_features, feat_save_path)
                    logger.info(
                        f"Saved pred feature with shape {final_features.shape} to {feat_save_path}"
                    )
                    del point_features, final_features

                if "ScanNetPP" in self.cfg.data.test.type:
                    # e.g. we want top-3 classes for each point
                    pred = pred.topk(3, dim=1)[1].cpu().numpy()  # shape => [N, 3]
                else:
                    # typical semantic seg => pick best
                    max_probs, argmax_indices = torch.max(pred, dim=1)
                    argmax_indices[max_probs < self.confidence_threshold] = ignore_index
                    pred = argmax_indices.cpu().numpy()

                if "origin_segment" in data_dict:
                    assert "inverse" in data_dict, (
                        "Inverse mapping is required to map pred to full origin_coord"
                    )
                    pred = pred[
                        data_dict["inverse"]
                    ]  # shape => [original_num_points, ...]
                    segment = data_dict["origin_segment"]
                if "pc_coord" in data_dict and "pc_segment" in data_dict:
                    segment = data_dict["pc_segment"]

                if self.pred_label_mapping is not None:
                    for key, item in self.pred_label_mapping.items():
                        pred[pred == key] = item

                # ================  Submission Saving ================
                os.makedirs(os.path.join(save_path, "submit"), exist_ok=True)
                if self.cfg.data.test.type in [
                    "ScanNetDataset",
                    "ScanNet200Dataset",
                    "ScanNetGSDataset",
                    "ScanNet200GSDataset",
                ]:
                    np.savetxt(
                        os.path.join(save_path, "submit", f"{data_name}.txt"),
                        self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                        fmt="%d",
                    )
                elif (
                    self.cfg.data.test.type == "ScanNetPPDataset"
                    or "ScanNetPP" in self.cfg.data.test.type
                ):
                    np.savetxt(
                        os.path.join(save_path, "submit", f"{data_name}.txt"),
                        pred.astype(np.int32),
                        delimiter=",",
                        fmt="%d",
                    )
                    pred = pred[:, 0] if pred.ndim > 1 else pred  # Handle 1D/2D pred
                elif self.cfg.data.test.type in [
                    "HoliCityGSDataset",
                    "Matterport3DGSDataset",
                ]:
                    np.savetxt(
                        os.path.join(save_path, "submit", f"{data_name}.txt"),
                        pred.astype(np.int32),
                        delimiter=",",
                        fmt="%d",
                    )
                elif self.cfg.data.test.type == "SemanticKITTIDataset":
                    sequence_name, frame_name = data_name.split("_")
                    os.makedirs(
                        os.path.join(
                            save_path,
                            "submit",
                            "sequences",
                            sequence_name,
                            "predictions",
                        ),
                        exist_ok=True,
                    )
                    submit = pred.astype(np.uint32)
                    submit = np.vectorize(
                        self.test_loader.dataset.learning_map_inv.__getitem__
                    )(submit).astype(np.uint32)
                    submit.tofile(
                        os.path.join(
                            save_path,
                            "submit",
                            "sequences",
                            sequence_name,
                            "predictions",
                            f"{frame_name}.label",
                        )
                    )
                elif self.cfg.data.test.type == "NuScenesDataset":
                    np.array(pred + 1).astype(np.uint8).tofile(
                        os.path.join(
                            save_path,
                            "submit",
                            "lidarseg",
                            "test",
                            f"{data_name}_lidarseg.bin",
                        )
                    )
            if self.skip_eval:
                continue
            # ---------------------------------------------------------------------
            # Apply neighbor voting if enabled
            if self.enable_voting:
                num_classes = self.num_classes
                ignore_index = self.ignore_index
                if "pc_coord" in data_dict and "pc_segment" in data_dict:
                    coords = data_dict["origin_coord"]
                    query_coords = data_dict["pc_coord"]
                    pred = neighbor_voting(
                        coords,
                        pred,
                        self.vote_k,
                        ignore_index,
                        num_classes,
                        valid_mask=data_dict.get("origin_feat_mask", None),
                        query_coords=query_coords,
                    )
                elif "origin_coord" in data_dict:
                    coords = data_dict["origin_coord"]
                    pred = neighbor_voting(
                        coords,
                        pred,
                        self.vote_k,
                        ignore_index,
                        num_classes,
                        valid_mask=data_dict.get("origin_feat_mask", None),
                    )
                else:
                    logger.warning(
                        "Neighbor voting requires 'origin_coord (3dgs)' or 'pc_coord (pc)' in data_dict, skipped.."
                    )
                if "origin_instance" in data_dict:
                    pred = clustering_voting(
                        pred, data_dict["origin_instance"], ignore_index
                    )

            np.save(pred_save_path, pred)
            intersection, union, target = intersection_and_union(
                pred, segment, self.num_classes, self.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            # Per‐scene IoU & accuracy
            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            # Running average across scenes so far
            mask_union = union_meter.sum != 0
            mask_target = target_meter.sum != 0
            m_iou = np.mean(
                (intersection_meter.sum / (union_meter.sum + 1e-10))[mask_union]
            )
            m_acc = np.mean(
                (intersection_meter.sum / (target_meter.sum + 1e-10))[mask_target]
            )

            batch_time.update(time.time() - end)
            logger.info(
                f"Test: {data_name} [{idx + 1}/{len(self.test_loader)}]-{segment.size} "
                f"Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Accuracy {acc:.4f} ({m_acc:.4f}) "
                f"mIoU {iou:.4f} ({m_iou:.4f})"
            )

        if self.skip_eval:
            logger.info(
                "<<<<<<<<<<<<<<<<< Tester End, Skipped Evaluation <<<<<<<<<<<<<<<<<"
            )
            return

        # ---------------------------------------------------------------------
        # Sync across processes if distributed
        # ---------------------------------------------------------------------
        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            # Merge results
            final_record = {}
            while record_sync:
                r = record_sync.pop()
                final_record.update(r)
                del r
            self._log_final_metrics(final_record, save_path)

    @staticmethod
    def collate_fn(batch):
        return batch

    def _log_final_metrics(self, final_record, save_path):
        logger = get_root_logger()

        intersection = np.sum(
            [v["intersection"] for v in final_record.values()], axis=0
        )
        union = np.sum([v["union"] for v in final_record.values()], axis=0)
        target = np.sum([v["target"] for v in final_record.values()], axis=0)

        iou_class = intersection / (union + 1e-10)
        accuracy_class = intersection / (target + 1e-10)

        mask_union = union != 0
        mask_target = target != 0
        mIoU = np.mean(iou_class[mask_union])
        mAcc = np.mean(accuracy_class[mask_target])
        allAcc = sum(intersection) / (sum(target) + 1e-10)

        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                mIoU, mAcc, allAcc
            )
        )

        # foreground metrics (excluding classes in excluded_classes)
        if self.excluded_indices:
            fg_iou_class = iou_class[self.keep_indices]
            fg_accuracy_class = accuracy_class[self.keep_indices]
            fg_mask_union = union[self.keep_indices] != 0
            fg_mask_target = target[self.keep_indices] != 0
            fg_mIoU = np.mean(fg_iou_class[fg_mask_union])
            fg_mAcc = np.mean(fg_accuracy_class[fg_mask_target])

            # foreground allAcc
            fg_intersection = intersection[self.keep_indices]
            fg_target = target[self.keep_indices]
            fg_allAcc = sum(fg_intersection) / (sum(fg_target) + 1e-10)

            logger.info(
                "Foreground Val result (excluding {} classes): mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    len(self.excluded_indices), fg_mIoU, fg_mAcc, fg_allAcc
                )
            )

        # Optionally log per‐class results
        if self.class_names:
            for i, cls_name in enumerate(self.class_names):
                logger.info(
                    f"Class_{i}-{cls_name} Result: iou/accuracy "
                    f"{iou_class[i]:.4f}/{accuracy_class[i]:.4f}"
                )
        else:
            for i in range(self.num_classes):
                logger.info(
                    f"Class_{i} iou/accuracy {iou_class[i]:.4f}/{accuracy_class[i]:.4f}"
                )

        # Save results to a text file
        result_file = os.path.join(save_path, "eval_results.txt")
        with open(result_file, "w") as f:
            f.write(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}\n".format(
                    mIoU, mAcc, allAcc
                )
            )

            # Write foreground metrics
            if self.excluded_indices:
                f.write(
                    "Foreground Val result (excluding {} classes): mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}\n".format(
                        len(self.excluded_indices), fg_mIoU, fg_mAcc, fg_allAcc
                    )
                )

            # Write per-class results
            f.write("\nPer-class results:\n")
            if self.class_names:
                for i, cls_name in enumerate(self.class_names):
                    f.write(
                        "Class_{}-{} Result: iou/accuracy {:.4f}/{:.4f}\n".format(
                            i, cls_name, iou_class[i], accuracy_class[i]
                        )
                    )
            else:
                for i in range(self.num_classes):
                    f.write(
                        "Class_{} iou/accuracy {:.4f}/{:.4f}\n".format(
                            i, iou_class[i], accuracy_class[i]
                        )
                    )

            # Mark which classes were excluded
            if self.excluded_indices:
                f.write("\nExcluded classes:\n")
                for idx in self.excluded_indices:
                    if hasattr(self.cfg.data, "names"):
                        f.write(f"Class_{idx}-{self.class_names[idx]}\n")
                    else:
                        f.write(f"Class_{idx}\n")

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")


@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>> SemSegTester Start Evaluation >>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
            or self.cfg.data.test.type == "ScanNetPPDataset"
            or "ScanNetPP" in self.cfg.data.test.type
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))  # for benchmark
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            print("data_name", data_name)
            print("origin_segment shape: ", data_dict["origin_segment"].size)
            # what is fragment_list?
            # scannetpp
            # print("data_dict", data_dict.keys()) # data_dict dict_keys(['origin_segment', 'inverse'])

            if os.path.isfile(
                pred_save_path
            ):  # if the file exists, load the pred and label only
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
                if "origin_segment" in data_dict.keys():
                    segment = data_dict["origin_segment"]
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                pred_coord = torch.zeros((segment.size, 3)).cuda()
                print("pred init shape: ", pred.shape)
                # pred_idx = torch,zeros((segment.size, 1)).cpu()
                for i in range(len(fragment_list)):  #  # From test-mode GridSample
                    fragment_batch_size = 1
                    s_i, e_i = (
                        i * fragment_batch_size,
                        min((i + 1) * fragment_batch_size, len(fragment_list)),
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    # print("input_dict shape: ", input_dict.keys()) # dict_keys(['coord', 'grid_coord', 'index', 'offset', 'feat'])
                    # print("input_dict[offset]", input_dict["offset"])
                    # print("feat", input_dict["feat"].shape)
                    # print("input_dict[coord]", input_dict["coord"].shape, input_dict["coord"].min(), input_dict["coord"].max())
                    # print("input_dict[grid_coord]", input_dict["grid_coord"].shape, input_dict["grid_coord"].min(), input_dict["grid_coord"].max())

                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(
                                non_blocking=True
                            )  # move to cuda

                    idx_part = input_dict["index"]
                    # index is get after grid sampler
                    with torch.no_grad():
                        pred_part = self.model(input_dict)[
                            "seg_logits"
                        ]  # (n, k) # Process each fragment
                        pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:  # iterate over the offset
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            # pred_idx[idx_part[bs:be]] = idx_part[bs:be]
                            pred_coord[idx_part[bs:be], :] = input_dict["coord"][bs:be]
                            # print("pred_part[bs:be]", pred_part[bs:be].shape)
                            # print("coord", input_dict["coord"][bs:be].shape)
                            # print("coord from input dict", input_dict["coord"][bs:be].min(), input_dict["coord"][bs:be].max())

                            bs = be

                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=i,
                            batch_num=len(fragment_list),
                        )
                    )
                if (
                    self.cfg.data.test.type == "ScanNetPPDataset"
                    or "ScanNetPP" in self.cfg.data.test.type
                ):
                    pred = pred.topk(3, dim=1)[1].data.cpu().numpy()
                    pred_coord = pred_coord.data.cpu().numpy()
                    # pred_idx = pred_idx.data.cpu().numpy()
                else:
                    pred = pred.max(1)[1].data.cpu().numpy()

                if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    pred = pred[data_dict["inverse"]]  # this can change the shape
                    pred_coord = pred_coord[data_dict["inverse"]]
                    # pred_idx = pred_idx[data_dict["inverse"]]
                    # print("pred shape after inverse operation: ", pred.shape)
                    segment = data_dict["origin_segment"]
                    # np.save(pred_save_path.replace('.npy','_coord.npy'), pred_coord)
                    print("pred shape after inverse operation: ", pred.shape)
                    print("segment shape after inverse operation: ", segment.shape)
                    # segment = np.frombuffer(segment) #segment.data.cpu().numpy()
                    # np.save(pred_save_path.replace('.npy','_segment.npy'), segment)

                print("pred shape: ", pred.shape)
                np.save(pred_save_path, pred)
            if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif (
                self.cfg.data.test.type == "ScanNetPPDataset"
                or "ScanNetPP" in self.cfg.data.test.type
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    pred.astype(np.int32),
                    delimiter=",",
                    fmt="%d",
                )
                pred = pred[:, 0]  # for mIoU, TODO: support top3 mIoU
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                submit = pred.astype(np.uint32)
                submit = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(submit).astype(np.uint32)
                submit.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            if comm.get_world_size() > 1:
                (
                    dist.all_reduce(intersection),
                    dist.all_reduce(union),
                    dist.all_reduce(target),
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            (
                intersection_meter.update(intersection),
                union_meter.update(union),
                target_meter.update(target),
            )

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class ClsVotingTester(TesterBase):
    def __init__(
        self,
        num_repeat=100,
        metric="allAcc",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_repeat = num_repeat
        self.metric = metric
        self.best_idx = 0
        self.best_record = None
        self.best_metric = 0

    def test(self):
        for i in range(self.num_repeat):
            logger = get_root_logger()
            logger.info(f">>>>>>>>>>>>>>>> Start Evaluation {i + 1} >>>>>>>>>>>>>>>>")
            record = self.test_once()
            if comm.is_main_process():
                if record[self.metric] > self.best_metric:
                    self.best_record = record
                    self.best_idx = i
                    self.best_metric = record[self.metric]
                info = f"Current best record is Evaluation {i + 1}: "
                for m in self.best_record.keys():
                    info += f"{m}: {self.best_record[m]:.4f} "
                logger.info(info)

    def test_once(self):
        logger = get_root_logger()
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        target_meter = AverageMeter()
        record = {}
        self.model.eval()

        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            voting_list = data_dict.pop("voting_list")
            category = data_dict.pop("category")
            data_name = data_dict.pop("name")
            # pred = torch.zeros([1, self.cfg.data.num_classes]).cuda()
            # for i in range(len(voting_list)):
            #     input_dict = voting_list[i]
            #     for key in input_dict.keys():
            #         if isinstance(input_dict[key], torch.Tensor):
            #             input_dict[key] = input_dict[key].cuda(non_blocking=True)
            #     with torch.no_grad():
            #         pred += F.softmax(self.model(input_dict)["cls_logits"], -1)
            input_dict = collate_fn(voting_list)
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                pred = F.softmax(self.model(input_dict)["cls_logits"], -1).sum(
                    0, keepdim=True
                )
            pred = pred.max(1)[1].cpu().numpy()
            intersection, union, target = intersection_and_union(
                pred, category, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            target_meter.update(target)
            record[data_name] = dict(intersection=intersection, target=target)
            acc = sum(intersection) / (sum(target) + 1e-10)
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
            accuracy_class = intersection / (target + 1e-10)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info("Val result: mAcc/allAcc {:.4f}/{:.4f}".format(mAcc, allAcc))
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        accuracy=accuracy_class[i],
                    )
                )
            return dict(mAcc=mAcc, allAcc=allAcc)

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class PartSegTester(TesterBase):
    def test(self):
        test_dataset = self.test_loader.dataset
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        num_categories = len(self.test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        self.model.eval()

        save_path = os.path.join(
            self.cfg.save_path, "result", "test_epoch{}".format(self.cfg.test_epoch)
        )
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, self.cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / self.cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = (
                    i * self.cfg.batch_size_test,
                    min((i + 1) * self.cfg.batch_size_test, len(data_dict_list)),
                )
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = self.model(input_dict)["cls_logits"]
                    pred_part = F.softmax(pred_part, -1)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, self.cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info(
                    "Test: {} {}/{}, Batch: {batch_idx}/{batch_num}".format(
                        data_name,
                        idx + 1,
                        len(test_dataset),
                        batch_idx=i,
                        batch_num=batch_num,
                    )
                )
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[category_index]
            parts_idx = self.test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} "
                "({batch_time.avg:.3f}) ".format(
                    data_name, idx + 1, len(self.test_loader), batch_time=batch_time
                )
            )

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info(
            "Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.".format(ins_mIoU, cat_mIoU)
        )
        for i in range(num_categories):
            logger.info(
                "Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}".format(
                    idx=i,
                    name=self.test_loader.dataset.categories[i],
                    iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                    iou_count=int(iou_count[i]),
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)
