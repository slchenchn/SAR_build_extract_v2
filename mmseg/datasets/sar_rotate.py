from .builder import DATASETS
from .npy_dataset import NpyDataset


import os.path as osp
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from torch.utils.data import Dataset

from mmseg.core import mean_iou
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class Sar_Rotate(NpyDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'Ground','Building')

    PALETTE = [[0, 0, 0],[255, 255, 255]]

    def __init__(self, **kwargs):
        super(Sar_Rotate, self).__init__(
            img_suffix='.npy',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
