'''
Author: Shuailin Chen
Created Date: 2021-08-28
Last Modified: 2021-08-28
	content: 
'''
import torch
from torch import nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .sep_aspp_head import DepthwiseSeparableASPPHead


@HEADS.register_module()
class Depthwise2SeparableASPPHead(DepthwiseSeparableASPPHead):
    ''' Atrous Separable Convolution decoder with two branches: 1)original classification head, 2)representation head. There two branch has identity structure, but not shape weights. This head mainly for ReCo algorithm

    Args:
        rep_channels (int): number of channels of representation head
        c1_channels (int): The intermediate channels of c1 decoder.
    '''

    def __init__(self, c1_in_channels, c1_channels, rep_channels=None, **kwargs):
        super().__init__(c1_in_channels, c1_channels, **kwargs)
        
        if not rep_channels:
            self.rep_channels = self.channels
        else:
            self.rep_channels = rep_channels

        self.rep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.rep_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.rep_channels,
                self.rep_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)

        representation = self.rep_bottleneck(output)

        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output, representation
        