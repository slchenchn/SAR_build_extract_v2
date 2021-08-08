from torch import nn
import torch


class SequentialMixBN(nn.Sequential):
    ''' Sequential nn class for mix BN 
    '''

    def forward(self, input, domain):
        for module in self:
            if 'bn' in str(type(module)):
                input = module(input, domain)
            else:
                input = module(input)   
        return input