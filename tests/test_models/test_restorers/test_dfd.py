import numpy as np
import pytest
import torch
from mmcv.runner import obj_from_dict
from mmcv.utils.config import Config

from mmedit.models.builder import build_model
from mmedit.datasets import SRFolderDataset


def test_dfd():

    model_cfg = dict(
        type='DFD',
        generator=dict(
            type='DFDNet',
            mid_channels=64),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        dictionary='/mnt/lustre/liyinshuo/01-git-clone/DFDNet/DictionaryCenter512',
        train_cfg=None,
        test_cfg=None,
        pretrained=None)

    train_cfg = None
    test_cfg = Config(dict(metrics=['PSNR', 'SSIM']))

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert restorer.__class__.__name__ == 'DFD'

    # prepare data
    
    inputs = torch.rand(1, 3, 512, 512)
    part_locations = dict(
        left_eye=torch.tensor([[146, 184, 225, 263]]),
        right_eye=torch.tensor([[283, 179, 374, 270]]),
        nose=torch.tensor([[229, 296, 282, 349]]),
        mouth=torch.tensor([[195, 305, 323, 433]]))
    targets = torch.rand(1, 3, 512, 512)
    data_batch = {'lq': inputs, 'gt': targets, 'location': part_locations}

    # test forward_test (cpu)
    outputs = restorer(**data_batch, test_mode=True)
    print(outputs.keys())
    print(outputs['output'].shape)
    return None

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        restorer = restorer.cuda()
        data_batch = {
            'lq': inputs.cuda(),
            'gt': targets.cuda(),
            'heatmap': heatmap.cuda()
        }

        # train_step
        optim_cfg = dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))
        generator = obj_from_dict(optim_cfg, torch.optim,
                                  dict(params=restorer.parameters()))
        discriminator = obj_from_dict(optim_cfg, torch.optim,
                                      dict(params=restorer.parameters()))
        optimizer = dict(generator=generator, discriminator=discriminator)
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['log_vars']['loss_pixel_v3'], float)
        assert outputs['num_samples'] == 1
        assert outputs['results']['lq'].shape == data_batch['lq'].shape
        assert outputs['results']['gt'].shape == data_batch['gt'].shape
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == (1, 3, 128, 128)

        # val_step
        data_batch.pop('heatmap')
        result = restorer.val_step(data_batch, meta=[{'gt_path': ''}])
        assert isinstance(result, dict)
        assert isinstance(result['eval_result'], dict)
        assert result['eval_result'].keys() == set({'PSNR', 'SSIM'})
        assert isinstance(result['eval_result']['PSNR'], np.float64)
        assert isinstance(result['eval_result']['SSIM'], np.float64)

        with pytest.raises(AssertionError):
            # evaluation with metrics must have gt images
            restorer(lq=inputs.cuda(), test_mode=True)

        with pytest.raises(TypeError):
            restorer.init_weights(pretrained=1)
        with pytest.raises(OSError):
            restorer.init_weights(pretrained='')


def test_dfd_all():

    test_pipeline = [
        dict(
            type='LoadImageFromFile',
            io_backend='disk',
            key='gt',
            flag='color',
            channel_order='rgb',
            backend='pillow'),
        dict(
            type='LoadImageFromFile',
            io_backend='disk',
            key='lq',
            flag='color',
            channel_order='rgb',
            backend='pillow'),
        dict(
            type='DetectFaceLandmark',
            image_key='lq',
            landmark_key='landmark',
            device='cuda'),
        dict(
            type='FacialFeaturesLocation',
            landmark_key='landmark',
            location_key='location'),
        # dict(
        #     type='Normalize',
        #     keys=['gt'],
        #     mean=[127.5, 127.5, 127.5],
        #     std=[127.5, 127.5, 127.5]),
        dict(dict(type='ImageToTensor', keys=['gt', 'lq']))
    ]
    sr_folder_dataset = SRFolderDataset(
        gt_folder='tests/data/dfd_gt',
        lq_folder='tests/data/dfd',
        scale=4,
        pipeline=test_pipeline)
    
    results = sr_folder_dataset.prepare_test_data(0)
    image = results['lq']
    location = results['location']
    data = np.load('/mnt/lustre/liyinshuo/01-git-clone/DFDNet/Imgs/RealSR/Step2_Landmarks/n000056_0060_01.png.npy', allow_pickle=True).item()
    image_load = data['image']
    location_load = data['locations']
    print(type(image), image.shape, image.min(), image.max())
    print(type(image_load), image_load.shape, image_load.min(), image_load.max())
    print(location)
    print(location_load)
    assert (image/127.5-1-image_load).max() < 1e-6
    print('data', (image/127.5-1-image_load).max())

    model_cfg = dict(
        type='DFD',
        generator=dict(
            type='DFDNet',
            mid_channels=64),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        dictionary='/mnt/lustre/liyinshuo/01-git-clone/DFDNet/DictionaryCenter512',
        train_cfg=None,
        test_cfg=None,
        pretrained=None)

    train_cfg = None
    test_cfg = dict(crop_border=0)

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    restorer.load_state_dict(torch.load('/mnt/lustre/liyinshuo/01-git-clone/DFDNet/user/dfd_net.pth'))

    # test forward_test (cpu)
    outputs = restorer.forward_test(
                     lq=image_load.unsqueeze(0),
                     gt=results['gt'],
                     location=location,
                     meta=[dict(gt_path='img.png')],
                     save_image=True,
                     save_path='work_dirs')
    print(outputs.keys())
    print(outputs['gt'].max(), outputs['gt'].min())
    print(outputs['output'].max(), outputs['output'].min())
    # pred = (outputs['output']*255).astype(np.uint8)
    gt = outputs['gt'].int()
    pred = (outputs['output']*255).int()
    print((gt-pred).abs().max())


if __name__ == '__main__':
    test_dfd_all()
