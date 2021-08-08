from mmcv.utils import Registry
from mmcv.cnn import NORM_LAYERS
from torch import nn
import torch
from torch.nn.modules.batchnorm import _BatchNorm, BatchNorm2d
import numpy as np
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional
from torch import Tensor, device, dtype


@NORM_LAYERS.register_module()
class MixBN(nn.Module):
    ''' Mix BN parameters 
    
    Args:
        ratio (float): mixup ratio, must in [0, 1]. Default: 0.5    
        detach (bool): if to detach BN with respect to target domain. Default: 
            True
        model (int): mix mode. 0: target domain's mean and var as source'
            domain's mean and var. Default: 0
    '''

    def __init__(self, num_features, ratio=0.5, detach=True, mode=0,
                momentum=0.9, eps=1e-5, **kargs):
        super().__init__(**kargs)
        assert ratio>=0 and ratio <=1
        self.ratio = ratio
        self.detach = detach
        self.mode = mode
        self.eps = eps
        self.momentum = momentum
 
        shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.running_mean = torch.zeros(shape)
        self.running_var = torch.ones(shape)       
        self.running_mean_dst = torch.zeros(shape)
        self.running_var_dst = torch.ones(shape)    
            
    def cuda(self, device=None):
        super().cuda(device=device)    
        self.gamma.cuda(device=device) 
        self.beta.cuda(device=device) 
        self.running_mean.cuda(device=device) 
        self.running_var.cuda(device=device) 
        self.running_mean_dst.cuda(device=device) 
        self.running_var_dst.cuda(device=device) 

    def forward(self, input, domain):
        '''
        Args:
            domain (list): list indicates source domain and target domain, 0
                indicates source, 1 indicates target            
        '''
        dst_idx = np.argwhere(domain).flatten()
        src_idx = np.argwhere(np.logical_not(domain)).flatten()
        dst_input = input[dst_idx, ...]
        src_input = input[src_idx, ...]
        
        # 
        dst_output, _, _, mean_dst, var_dst = mix_bn(
            dst_input, self.gamma, self.beta, self.running_mean_dst, self.running_var_dst,
            self.eps, self.momentum, ratio=1, mean=0, var=0)

        if self.detach:
            mean_dst = mean_dst.detach()
            var_dst = var_dst.detach()

        if self.mode==0:
            ''' replace source mean and var with targe mean and var'''
            src_ouput, self.running_mean, self.running_var, _, _ = mix_bn(src_input, self.gamma, self.beta, self.running_mean, self.running_var, self.eps, self.momentum, self.ratio, mean_dst, var_dst)
        else:
            raise NotImplementedError
        
        output = torch.empty_like(input)
        output[dst_idx, ...] = dst_output
        output[src_idx, ...] = src_ouput
        
        return src_ouput



def mix_bn(X, gamma, beta, running_mean, running_var, eps, momentum, ratio,
            mean=0, var=0):
    ''' BatchNorm implementation from dive into deep learning
    '''

    # Use `is_grad_enabled` to determine whether the current mode is training
    # mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance
        # obtained by running average
        X_hat = (X - running_mean) / torch.sqrt(running_var + eps)
    else:
        local_mean = X.mean(dim=(0, 2, 3), keepdim=True)
        local_var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)

        mean = ratio * local_mean + (1-ratio)*mean
        var = ratio * local_var + (1-ratio)*var

        X_hat = (X - mean) / torch.sqrt(var + eps)

        # Update the mean and variance using running average
        running_mean = momentum * running_mean + (1.0 - momentum) * mean
        running_var = momentum * running_var + (1.0 - momentum) * var

    Y = gamma * X_hat + beta # Scale and shift

    return Y, running_mean.data, running_var.data, mean, var

