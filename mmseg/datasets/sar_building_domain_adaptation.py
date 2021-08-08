'''
Author: Shuailin Chen
Created Date: 2021-07-12
Last Modified: 2021-08-04
	content: 
'''
from .builder import DATASETS
from .domain_adaptation import DomainAdaptationDataset


@DATASETS.register_module()
class SARBuildingDomainAdaptation(DomainAdaptationDataset):
    """SAR building dataset for domain adaptation purpose

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'Ground','Building')

    PALETTE = [[0, 0, 0],[255, 255, 255]]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
