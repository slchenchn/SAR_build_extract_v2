'''
Author: Shuailin Chen
Created Date: 2021-08-10
Last Modified: 2021-08-10
	content: 
'''
import torch
from torch import nn
from mmcv.cnn import NORM_LAYERS

from ..segmentors import Semi



@NORM_LAYERS.register_module()
class TestMixBN(nn.BatchNorm2d):

    def forward(self, input, domain=None):

        src_input, dst_input = Semi.split_domins_data(input, domain=domain)

        src_output = super().forward(src_input)

        
        return 