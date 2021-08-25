import numbers
import numpy as np
import os
import os.path as osp

import mmcv
import torch

from mmedit.core import tensor2img
from mmedit.models.common import ImgNormalize
from ..builder import build_backbone, build_component, build_loss
from ..common import set_requires_grad
from ..registry import MODELS
from .basic_restorer import BasicRestorer


def get_sum(x, axis=None, keepdim=False):
    """Get sum of tensor.

    Args:
        x (Tensor): Input tensor.
        axis (Tuple[int] | None): Calculated dimensions.
    
    returns:
        Tensor: Output tensor.
    """

    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


@MODELS.register_module()
class DFD(BasicRestorer):
    """DFD model for Face Super-Resolution.

    Paper: Blind Face Restoration via Deep Multi-scale Component Dictionaries.

    Args:
        generator (dict): Config for the generator.
        pixel_loss (dict): Config for the pixel loss.
        train_cfg (dict): Config for train. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 dictionary=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(BasicRestorer, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # get dictionary
        self.dictionary = dict()
        self.get_dictionary(dictionary)

        # model
        # generator['dictionary'] = self.dictionary
        self.generator = build_backbone(generator)
        self.img_normalize = ImgNormalize(
            pixel_range=1,
            img_mean=(127.5, 127.5, 127.5),
            img_std=(127.5, 127.5, 127.5))
        self.img_denormalize = ImgNormalize(
            pixel_range=1,
            img_mean=(1, 1, 1),
            img_std=(2, 2, 2),
            sign=-1)

        # loss
        self.pixel_loss = build_loss(pixel_loss)

        # pretrained
        self.init_weights(pretrained)

        # fix pre-trained networks
        self.register_buffer('step_counter', torch.zeros(1))

    def get_dictionary(self, dictionary):
        """Get dictionary from files, saved in self.dictionary.

        Args:
            dictionary (str): Dictionary path.
        """
        
        parts = ['left_eye','right_eye','nose','mouth']        
        part_sizes = np.array([80,80,50,110])
        channel_sizes = np.array([128,256,512,512])

        for j, size in enumerate([256, 128, 64, 32]):
            self.dictionary[size] = dict()
            for i, part in enumerate(parts):
                path = os.path.join(dictionary, f'{part}_{size}_center.npy')
                tensor = torch.from_numpy(np.load(path, allow_pickle=True))
                self.dictionary[size][part] = tensor.reshape(
                    tensor.shape[0],
                    channel_sizes[j],
                    part_sizes[i]//(2**(j+1)),
                    part_sizes[i]//(2**(j+1)))
                print(self.dictionary[size][part].shape)

    def forward(self, lq, gt=None, test_mode=False, location=None, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt=gt, location=location, **kwargs)

        return self.generator.forward(lq, location, self.dictionary)

    def forward_test(self,
                     lq,
                     gt=None,
                     location=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ image.
            gt (Tensor): GT image.
            meta (list[dict]): Meta data, such as path of GT file.
                Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results, which contain either key(s)
                1. 'eval_result'.
                2. 'lq', 'pred'.
                3. 'lq', 'pred', 'gt'.
        """

        # generator
        with torch.no_grad():
            # self.img_normalize = ImgNormalize(
            #     pixel_range=1,
            #     img_mean=(127.5, 127.5, 127.5),
            #     img_std=(127.5, 127.5, 127.5))
            # self.img_denormalize = ImgNormalize(
            #     pixel_range=1,
            #     img_mean=(1, 1, 1),
            #     img_std=(2, 2, 2),
            #     sign=1)
            print('lq_pred', lq.shape, lq.min(), lq.max())
            # lq = self.img_normalize(lq)
            print('lq', lq.shape, lq.min(), lq.max())
            pred = self.generator.forward(lq, location, self.dictionary)
            # print('pred', pred.shape, pred.min(), pred.max())
            pred = self.img_denormalize(pred)
            print('out', pred.shape, pred.min(), pred.max())

            # if gt is not None:
            #     gt = self.img_denormalize(gt)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(pred, gt))
        else:
            results = dict(lq=lq.cpu(), output=pred.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            if 'gt_path' in meta[0]:
                pred_path = meta[0]['gt_path']
            else:
                pred_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(pred_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(pred), save_path)
        # torch.save(self.state_dict(), 'work_dirs/dfd/model2.pth')
        return results

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            self.generator.init_weights(pretrained, strict)
            if self.discriminator:
                self.discriminator.init_weights(pretrained, strict)
        elif pretrained is not None:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data, which requires
                'lq', 'gt'
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output, which includes:
                log_vars, num_samples, results (lq, gt and pred).

        """
        # data
        lq = data_batch['lq']
        gt = data_batch['gt']

        # generate
        pred = self(**data_batch, test_mode=False)

        # loss
        losses = dict()
        log_vars = dict()
        losses['loss_pixel'] = self.pixel_loss(pred, gt)
        
        loss, log_vars = self.parse_losses(losses)
        log_vars.update(log_vars)

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=pred.cpu()))

        self.step_counter += 1

        return outputs
