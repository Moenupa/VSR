exp_name = 'ganbasicvsr_spynet_c64b20_1x30x8_lr5e-5_30k_stm3k'

scale = 4

# model settings
model = dict(
    type='GANBasicVSR',
    generator=dict(
        type='GANBasicVSRNet',
        mid_channels=64,
        num_propagation_blocks=20,
        srgan_model=dict(
            generator=dict(
                type='RRDBNet',
                in_channels=3,
                out_channels=3,
                mid_channels=64,
                num_blocks=23,
                growth_channels=32,
                upscale_factor=4),
            discriminator=dict(type='ModifiedVGG', in_channels=3, mid_channels=64),
            pixel_loss=dict(type='L1Loss', loss_weight=1e-2, reduction='mean'),
            perceptual_loss=dict(
                type='PerceptualLoss',
                layer_weights={'34': 1.0},
                vgg_type='vgg19',
                perceptual_weight=1.0,
                style_weight=0,
                norm_img=False),
            gan_loss=dict(
                type='GANLoss',
                gan_type='vanilla',
                loss_weight=5e-3,
                real_label_val=1.0,
                fake_label_val=0),
        ),
        srgan_checkpoint='work_dirs/esrgan_x4c64b23g32_g1_100k_stm3k/iter_40000.pth',
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth',
    ),
    discriminator=dict(
        type='UNetDiscriminatorWithSpectralNorm',
        in_channels=3,
        mid_channels=64,
        skip_connection=True),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={
            '2': 0.1,
            '7': 0.1,
            '16': 1.0,
            '25': 1.0,
            '34': 1.0,
        },
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=5e-2,
        real_label_val=1.0,
        fake_label_val=0),
    is_use_sharpened_gt_in_pixel=True,
    is_use_sharpened_gt_in_percep=True,
    is_use_sharpened_gt_in_gan=False,
    is_use_ema=True,

)

# model training and testing settings
train_cfg = dict()
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)  # change to [] for test

# dataset settings
train_dataset_type = 'SRFolderMultipleGTDataset'
val_dataset_type = 'SRFolderMultipleGTDataset'
test_dataset_type = 'SRFolderMultipleGTDataset'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

val_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

test_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:08d}.png'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),  # 8 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=100,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/STM/train/lq',
            gt_folder='data/STM/train/gt',
            num_input_frames=5,
            pipeline=train_pipeline,
            scale=4,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='data/STM/val/lq',
        gt_folder='data/STM/val/gt',
        num_input_frames=5,
        pipeline=val_pipeline,
        scale=4,
        test_mode=False),
    # test
    test=dict(
        type=test_dataset_type,
        lq_folder='data/STM3k/test30/lq',
        gt_folder='data/STM3k/test30/gt',
        num_input_frames=100,
        pipeline=val_pipeline,
        scale=4,
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=5e-5, betas=(0.9, 0.99)),
    discriminator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)))

# learning policy
total_iters = 40000
lr_config = dict(policy='Step', by_epoch=False, step=[300], gamma=1)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)

# remove gpu_collect=True in non distributed training
evaluation = dict(interval=5000, save_image=False)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# custom hook
custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        interp_cfg=dict(momentum=0.999),
    )
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./out/{exp_name}'
load_from = ''  # noqa
resume_from = None
workflow = [('train', 1)]
