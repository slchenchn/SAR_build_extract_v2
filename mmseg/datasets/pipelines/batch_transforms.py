'''
Author: Shuailin Chen
Created Date: 2021-09-01
Last Modified: 2021-09-02
	content: perform image transformation on a batch of image, use the interfaces of MMSeg
'''

import torch
from torch import Tensor
from torchvision.transforms import Normalize
import cv2

from mmseg.utils import visualize_multiple_images
from .transforms import PhotoMetricDistortion
from .formating import to_tensor
from ..builder import PIPELINES


@PIPELINES.register_module()
class BatchPhotoMetricDistortion(PhotoMetricDistortion):
    """Apply photometric distortion to a batch of images

    Args:
        to_tensor (bool): if transform to pytorch Tensor. Default: True
    """

    def __init__(self, *args, to_tensor=True, **kargs):
        super().__init__(*args, **kargs)
        self.to_tensor = to_tensor

    def __call__(self, batch_result):
        imgs = batch_result['img']
        assert imgs.ndim == 4

        new_imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1)
        for ii, img in enumerate(new_imgs):
            new_imgs[ii, ...] = super().__call__({'img': img})['img']
        
        new_imgs = new_imgs.transpose(0, 3, 1, 2)

        if self.to_tensor:
            new_imgs = to_tensor(new_imgs)

        batch_result['img'] = new_imgs
        return batch_result


@PIPELINES.register_module()
class BatchNormalize(Normalize):
    ''' Normalize of Batched data in favor of MMSeg's interface 
    
    Args:
        to_rgb (bool): Whether to convert the image from BGR to RGB
    '''
    def __init__(self, to_rgb, inplace=False, **kargs):
        super().__init__(inplace=inplace, **kargs)
        self.to_rgb = to_rgb

    def __call__(self, batch_result: dict) -> Tensor:
        imgs = batch_result['img']
        if self.to_rgb:
            imgs = imgs[:, [2, 1, 0], ...]

        batch_result['img'] = super().forward(imgs.float())
        return batch_result


@PIPELINES.register_module()
class VisualizeMultiImages():
    ''' Visualize multiple images from a concatenated images file, mainly used in the pipeline

    Args:
        dst_path (str): destination path to save images
        channel_per_image (int): channels per image. Default:3
    '''

    def __init__(self, dst_path, channel_per_image=3) -> None:
        self.dst_path = dst_path
        self.channel_per_image = channel_per_image

    def __call__(self, batch_result: dict):
        imgs = batch_result['img']
        visualize_multiple_images(imgs, self.dst_path,
                                        self.channel_per_image)

        return batch_result
