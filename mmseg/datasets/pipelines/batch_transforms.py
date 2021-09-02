'''
Author: Shuailin Chen
Created Date: 2021-09-01
Last Modified: 2021-09-01
	content: perform image transformation on a batch of image
'''

import torch
from torch import Tensor

from .transforms import PhotoMetricDistortion
from ..builder import PIPELINES


@PIPELINES.register_module()
class BatchPhotoMetricDistortion(PhotoMetricDistortion):
    """Apply photometric distortion to a batch of images

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def __call__(self, imgs: Tensor):

        assert imgs.ndim == 4
        new_imgs = imgs.cpu().numpy()
        for ii, img in enumerate(new_imgs):
            new_imgs[ii, ...] = super().__call__({'img': img})['img']
        
        return new_imgs


# @PIPELINES.register_module()
# class BatchNormalize()


