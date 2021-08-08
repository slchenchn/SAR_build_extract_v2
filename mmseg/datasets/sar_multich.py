from .builder import DATASETS
from .npy_dataset import NpyDataset


@DATASETS.register_module()
class Sar_multich(NpyDataset):
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
        super(Sar_multich, self).__init__(
            img_suffix='.npy',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
