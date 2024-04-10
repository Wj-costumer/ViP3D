_base_ = [
    './_base_/nus-3d.py',
    './_base_/default_runtime.py'
]
workflow = [('train', 1)]
plugin = True
plugin_dir = 'plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names = [
    'car', 'truck', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian',
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='ViP3D',
    use_grid_mask=True,  # use grid mask
    num_classes=7,
    num_query=300,
    bbox_coder=dict(
        type='DETRTrack3DCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_num=100,
        num_classes=7),
    fix_feats=False,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    qim_args=dict(
        qim_type='QIMBase',
        merger_dropout=0, update_query_pos=True,
        fp_ratio=0.3, random_drop=0.1),
    mem_cfg=dict(
        memory_bank_type='MemoryBank',
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
    ),
    radar_encoder=dict(
        type='RadarPointEncoderXY',
        in_channels=13,
        out_channels=[32],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01), ),
    img_backbone=dict(
        type='ResNet',
        with_cp=False,
        # with_cp=True,
        # pretrained='open-mmlab://detectron2/resnet50_caffe',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    loss_cfg=dict(
        type='ClipMatcher',
        num_classes=7,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3DTrack',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            pc_range=point_cloud_range),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=4,
        norm_cfg=dict(type='BN2d'),
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='DeformableDETR3DCamHeadTrackPlusRaw',
        num_classes=7,
        in_channels=256,
        num_cams=6,
        num_feature_levels=4,
        with_box_refine=True,
        transformer=dict(
            type='Detr3DCamTrackTransformer',
            decoder=dict(
                type='Detr3DCamTrackPlusTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='Detr3DCrossAtten',
                            pc_range=point_cloud_range,
                            num_points=1,
                            embed_dims=256,
                        )
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        pc_range=point_cloud_range,
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5), ),
    do_pred=True,
    relative_pred=True,
    agents_layer_0=True,
    add_branch=True,
    predictor=dict(
        hidden_size=128,
        laneGCN=True,
        decoder=dict(
            variety_loss=True,
            variety_loss_prob=True,
            hidden_size=128,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='GIoU3DCost', weight=0.0),
                pc_range=point_cloud_range)
        )
    ),
)

# x y z rcs vx vy vx_comp vy_comp x_rms y_rms vx_rms vy_rms
radar_use_dims = [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 16, 17, 18]
dataset_type = 'NuScenesTrackDatasetRadar'
data_root = 'data/nuscenes/'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=1,
        use_dim=radar_use_dims,
        max_num=100, ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='InstanceRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32)
]

train_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'radar',
                                 'pred_matrix', 'polyline_spans', 'mapping', 'instance_idx_2_labels'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=1,
        use_dim=radar_use_dims,
        max_num=100, ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]

test_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=['points', 'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'radar',
                                 'pred_matrix', 'polyline_spans', 'mapping', 'instance_idx_2_labels'])
    # dict(type='Collect3D', keys=['points', 'img',])
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        num_frames_per_sample=3,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_tracking_infos_train.pkl',
        pipeline_single=train_pipeline,
        pipeline_post=train_pipeline_post,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        do_pred=True),
    # ),
    val=dict(type=dataset_type, pipeline_single=test_pipeline, pipeline_post=test_pipeline_post, classes=class_names, modality=input_modality,
             ann_file=data_root + 'nuscenes_tracking_infos_val.pkl',
             num_frames_per_sample=1,
             do_pred=True),
    test=dict(type=dataset_type, pipeline_single=test_pipeline,
              pipeline_post=test_pipeline_post,
              classes=class_names, modality=input_modality,
              ann_file=data_root + 'nuscenes_tracking_infos_val.pkl',
              num_frames_per_sample=1,
              do_pred=True))

optimizer = dict(
    type='AdamW',
    # type='SGD',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 24
evaluation = dict(interval=24)

runner = dict(type='EpochBasedRunner', max_epochs=24)

find_unused_parameters = True
load_from = 'ckpts/detr3d_resnet50.pth'

fp16 = dict(loss_scale='dynamic')
