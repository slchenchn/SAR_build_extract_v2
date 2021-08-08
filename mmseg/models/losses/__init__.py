'''
Author: Shuailin Chen
Created Date: 2021-07-12
Last Modified: 2021-07-14
	content: 
'''
from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .dice_loss import DiceLoss
from .lovasz_loss import LovaszLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss

from .whitening import RelaxedInstanceWhiteningLoss
from .cross_entropy_loss_semi import CrossEntropyLossSemi


# __all__ = [
#     'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
#     'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
#     'weight_reduce_loss', 'weighted_loss', 'LovaszLoss', 'DiceLoss'
# ]
