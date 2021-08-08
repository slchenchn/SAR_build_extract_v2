'''
Author: Shuailin Chen
Created Date: 2021-07-14
Last Modified: 2021-07-14
	content: 
'''
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .fcn_head import FCNHead


@HEADS.register_module()
class NewFCNHead(FCNHead):
    """ new FCN head, enable truly identity operation, mainly for calculate loss. the original FCN head will have a conv_seg layer even its num_convs=0, thus not a truly identity head.

    Args:
        see FCN head for detail, besides, when num_convs=0, it performs truly identity mapping
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        """overload Forward function, when num_convs=0, it performs truly identity mapping"""

        x = self._transform_inputs(inputs)
        output = x
        if self.num_convs != 0:
            output = self.convs(x)
            if self.concat_input:
                output = self.conv_cat(torch.cat([x, output], dim=1))

            output = self.cls_seg(output)

        return output
