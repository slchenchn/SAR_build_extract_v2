'''
Author: Shuailin Chen
Created Date: 2021-09-03
Last Modified: 2021-09-03
	content: semi-supervised framwork with pixel-level contrast
'''

from copy import deepcopy
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from torchvision.transforms.functional import normalize
import mylib.image_utils as iu

from mmseg.ops import resize
from mmseg.core import add_prefix
from mmseg.datasets.pipelines import Compose
from ..builder import SEGMENTORS
from .semi_v2 import SemiV2


@SEGMENTORS.register_module()
class PiCo(SemiV2):
    ''' pixel-level contrast algorithm for semi-supervised semantic segmentation

    Args:
        momentum (float): momentum to update the mean teacher model, must
            between [0, 1]. Default: 0.99
        strong_thres (float): strong threshold to filter the difficult
            samples, must between [0, 1]. Default: 0.97
        weak_thres (float): weak threshold to filter the unsure
            samples, must between [0, 1]. Default: 0.7
        tmperature (float): tmperature in the contrastive loss. Default: 0.5
        apply_pseudo_loss (bool): whether to apply pseudo labeling loss. 
            Default: True
        unlabeled_aug (list[dict]): Processing pipeline for unlabled data
    '''

    def __init__(self,
                momentum = 0.99,
                strong_thres = 0.97,
                weak_thres = 0.7,
                tmperature = 0.5,
                apply_pseudo_loss=True,
                unlabeled_aug=None,
                **kargs):
        super().__init__(**kargs)
        self.momentum = momentum
        self.strong_thres = strong_thres
        self.weak_thres = weak_thres
        self.tmperature = tmperature
        self.apply_pseudo_loss = apply_pseudo_loss
        if unlabeled_aug is not None:
            self.unlabeled_aug = {k:Compose(v) 
                                for k, v in unlabeled_aug.items()}
        else:
             self.unlabeled_aug = None

        # for updating EMA model
        self.step = 0
