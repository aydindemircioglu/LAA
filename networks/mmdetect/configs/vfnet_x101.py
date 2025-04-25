# The new config inherits a base config to highlight the necessary modification
_base_ = [
    '../../mmdetection/configs/vfnet/vfnet_x101-64x4d-mdconv-c3-c5_fpn_ms-2x_coco.py',
]



data = dict(
    workers_per_gpu=8,
)

# Modify dataset related settings
metainfo = {
    'classes': ('Heart', ),
    'palette': [
        (220, 20, 60),
    ]
}

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=".",
        metainfo=metainfo,
        ann_file='train/annotation_coco.json',
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomCrop', crop_size=(480, 480)),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='PhotoMetricDistortion', brightness_delta=16,  contrast_range=(0.8, 1.2), saturation_range=(0.8, 1.2),  hue_delta=0),
            dict(type='Corrupt', corruption = "gaussian_noise"),
            dict(type='Corrupt', corruption = "elastic_transform"),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ],
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=".",
        metainfo=metainfo,
        ann_file='val/annotation_coco.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

# Modify metric related settings
#val_evaluator = dict(ann_file=data_root + 'val/annotation_coco.json')
#test_evaluator = val_evaluator

load_from = '/home/aydin/uke/mmdetection/checkpoints/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-b5f6da5e.pth'





#
