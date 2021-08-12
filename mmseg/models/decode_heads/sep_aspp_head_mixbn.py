'''
Author: Shuailin Chen
Created Date: 2021-08-08
Last Modified: 2021-08-12
	content: 
'''
import torch
import torch.nn as nn
from ..layers import ConvModuleMixBN, DepthwiseSeparableConvModuleMixBN, SequentialMixBN
from ..segmentors import Semi

from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head_mixbn import ASPPHeadMixBN, ASPPModuleMixBN


class DepthwiseSeparableASPPModuleMixBN(ASPPModuleMixBN):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv for domain adaptation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModuleMixBN(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


@HEADS.register_module()
class DepthwiseSeparableASPPHeadMixBN(ASPPHeadMixBN):
    """Encoder-Decoder with Atrous Separable Convolution for domain adaptation

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super().__init__(**kwargs)
        assert c1_in_channels >= 0
        self.aspp_modules = DepthwiseSeparableASPPModuleMixBN(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModuleMixBN(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = SequentialMixBN(
            DepthwiseSeparableConvModuleMixBN(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModuleMixBN(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, inputs, domain=None):
        x = self._transform_inputs(inputs)

        domain = Semi.check_domain(data=x, domain=domain)

        aspp_outs = [
            resize(
                self.image_pool(x, domain=domain),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x, domain=domain))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs, domain=domain)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0], domain=domain)
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output, domain=domain)
        output = self.cls_seg(output)
        return output
