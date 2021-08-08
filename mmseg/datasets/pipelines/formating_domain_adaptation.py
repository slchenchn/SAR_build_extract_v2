'''
Author: Shuailin Chen
Created Date: 2021-08-04
Last Modified: 2021-08-04
	content: formatting for domain adaptation, deprecated.
'''

from collections.abc import Sequence

import mmcv.runner.base_runner
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES
from .formating import Collect



@PIPELINES.register_module()
class CollectDA(Collect):
    """Collect function for domain adaptation purpose
    """

    def __init__(self, **kargs):
        super().__init__(**kargs)

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'
