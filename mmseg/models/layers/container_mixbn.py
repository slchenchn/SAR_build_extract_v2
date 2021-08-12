'''
Author: Shuailin Chen
Created Date: 2021-08-08
Last Modified: 2021-08-12
	content: 
'''

from torch import nn


class SequentialMixBN(nn.Sequential):
    ''' Sequential nn class for mix BN 
    '''

    def forward(self, input, domain):
        for module in self:
            if 'bn' in str(type(module)).lower():
                input = module(input, domain=domain)
            else:
                input = module(input)   
        return input