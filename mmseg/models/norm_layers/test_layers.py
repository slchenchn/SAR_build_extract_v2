'''
Author: Shuailin Chen
Created Date: 2021-08-10
Last Modified: 2021-08-15
	content: 
'''
import torch
from torch import nn
from mmcv.cnn import NORM_LAYERS
from torch.nn import Identity

from ..segmentors import Semi



@NORM_LAYERS.register_module()
class TestMixBN(nn.BatchNorm2d):
    ''' Mimic the original BatchNorm2D layer
    '''
    def forward(self, input, domain):

        src_input, dst_input = Semi.split_domins_data(input, domain=domain)

        src_output = super().forward(src_input)

        output = Semi.merge_domains_data((src_output, dst_input), domain=domain)
        return output


@NORM_LAYERS.register_module()
class NormIdentity(Identity):
    ''' Just a identity layer, compatible with domain adaptation framwork
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, domain=None):
        return super().forward(input)