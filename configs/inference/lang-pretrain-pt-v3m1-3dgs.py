# inference config for SceneSplat LangPretrainer

model = dict(
    type="LangPretrainer",
    backbone=dict(
        type="PT-v3m1",
        in_channels=11,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2),
        enc_depths=(2, 2, 2, 6),
        enc_channels=(32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2),
        dec_channels=(768, 512, 256),
        dec_num_head=(16, 16, 16),
        dec_patch_size=(1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CosineSimilarity", reduction="mean", loss_weight=1.0),
        dict(type="L2Loss", reduction="mean", loss_weight=1.0),
        dict(
            type="AggregatedContrastiveLoss",
            temperature=0.2,
            reduction="mean",
            loss_weight=0.02,
            schedule="last_75",
        ),
    ],
)

feat_keys = ("color", "opacity", "quat", "scale")
grid_sample_keys = (
    "coord",
    "color",
    "opacity",
    "quat",
    "scale",
    "segment",
    "valid_feat_mask",
)
grid_sample_keys_test = (
    "coord",
    "color",
    "opacity",
    "quat",
    "scale",
    "segment",
    "valid_feat_mask",
)
collect_keys_test = (
    "coord",
    "grid_coord",
    "index",
    "segment",
    "valid_feat_mask",
    "pc_coord",
    "pc_segment",
)

inference = dict(
    transform=[
        dict(type="CenterShift", apply_z=True),
        dict(type="NormalizeColor"),
        dict(
            type="Copy",
            keys_dict=dict(
                segment="origin_segment",
                coord="origin_coord",
                valid_feat_mask="origin_feat_mask",
            ),
        ),
        dict(
            type="GridSample",
            grid_size=0.01,
            hash_type="fnv",
            mode="train",
            keys=grid_sample_keys,
            apply_to_pc=False,
            return_inverse=True,
        ),
    ],
    test_cfg=dict(
        voxelize=dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="test",
            keys=grid_sample_keys_test,
            apply_to_pc=False,
            return_grid_coord=True,
        ),
        crop=None,
        post_transform=[
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=collect_keys_test,
                feat_keys=feat_keys,
            ),
        ],
        aug_transform=[
            [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[0],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                )
            ]
        ],
    ),
    chunk_size=600000,
    save_features=dict(
        output_dir=None,
        backbone=dict(enabled=True, file_name="feat.pt"),
    ),
    default_scene_name="scenesplat_scene",
    device="cuda",
    return_numpy=True,
)
