''' deprecated '''

import numpy as np

from ..builder import LOSSES
from .cross_entropy_loss import CrossEntropyLoss


@LOSSES.register_module()
class CrossEntropyLossSemi(CrossEntropyLoss):
    ''' Cross entropy loss for semi supervised task, where no loss is calculated for samples without labels
    '''

    def forward(self, cls_score, label, *args, has_labels=None, **kargs):
        
        src_idx = np.argwhere(np.logical_not(has_labels)).flatten()
        src_cls_score = cls_score[src_idx, ...]
        src_label = label[src_idx, ...]

        return super().forward(src_cls_score, label, *args, **kargs)