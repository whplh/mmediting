exp_name = 'dfd_try'
scale = 1

# model settings
model = dict(
    type='DFD',
    generator=dict(
        type='DFDNet',
        mid_channels=64),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    dictionary='/mnt/lustre/liyinshuo/01-git-clone/DFDNet/DictionaryCenter512',
    train_cfg=None,
    test_cfg=None,
    pretrained=None)
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'])

# dataset settings
val_dataset_type = 'SRFolderDataset'
valid_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb',
        backend='cv2'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb',
        backend='cv2'),
    dict(
        type='DetectFaceLandmark',
        image_key='lq',
        landmark_key='landmark',
        device='cuda'),
    dict(
        type='FacialFeaturesLocation',
        landmark_key='landmark',
        location_key='location'),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'location'],
        meta_keys=['gt_path', 'lq_path'])
]

data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=60,
        dataset=dict(
            type=val_dataset_type,
            gt_folder='tests/data/dfd_gt',
            lq_folder='tests/data/dfd',
            pipeline=valid_pipeline,
            scale=scale)),
    val=dict(
        type=val_dataset_type,
        gt_folder='tests/data/dfd_gt',
        lq_folder='tests/data/dfd',
        pipeline=valid_pipeline,
        scale=scale),
    test=dict(
        type=val_dataset_type,
        gt_folder='tests/data/dfd_gt',
        lq_folder='tests/data/dfd',
        pipeline=valid_pipeline,
        scale=scale))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1.e-4))

# learning policy
total_iters = 2000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[10000, 20000, 40000, 80000],
    gamma=0.5)

checkpoint_config = dict(interval=1, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=2000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
