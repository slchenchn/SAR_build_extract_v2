'''
Author: Shuailin Chen
Created Date: 2021-07-12
Last Modified: 2021-08-10
	content: 
'''
import torch
import torch.nn as nn
from ..layers import ConvModuleMixBN, SequentialMixBN
from ..segmentors import Semi

from ..builder import HEADS
from .decode_head_mixbn import BaseDecodeHeadMixBN


@HEADS.register_module()
class FCNHeadMixBN(BaseDecodeHeadMixBN):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super().__init__(**kwargs)

        # if num_convs == 0:
        #     assert self.in_channels == self.channels


        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            conv_padding = (kernel_size // 2) * dilation
            convs = []
            convs.append(
                ConvModuleMixBN(
                    self.in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            for i in range(num_convs - 1):
                convs.append(
                    ConvModuleMixBN(
                        self.channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=conv_padding,
                        dilation=dilation,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.convs = SequentialMixBN(*convs)
        if self.concat_input:
            self.conv_cat = ConvModuleMixBN(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs, domain=None):
        x = self._transform_inputs(inputs)

        domain = Semi.check_domain(data=x, domain=domain)

        output = self.convs(x, domain=domain)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1), domain=domain)
        output = self.cls_seg(output)
        return output
