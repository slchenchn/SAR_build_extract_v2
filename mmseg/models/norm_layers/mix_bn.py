'''
Author: Shuailin Chen
Created Date: 2021-08-08
Last Modified: 2021-08-14
	content: 
'''

from mmcv.cnn import NORM_LAYERS
import torch.nn.functional as F
import torch
from torch.nn.modules.batchnorm import _BatchNorm, BatchNorm2d
from torch import Tensor
from typing import Optional
from mmcv.utils import print_log


from ..segmentors import Semi


@NORM_LAYERS.register_module()
class MixBN(BatchNorm2d):
    ''' Mix BN parameters 
    
    Args:
        ratio (float): mixup ratio, must in [0, 1]. Default: 0.5    
        detach (bool): if to detach BN with respect to target domain. Default: 
            True
        model (int): mix mode. 0: target domain's mean and var as source'
            domain's mean and var; 1: target domain's mean and var as source'
            domain's bias and weight, while in test mode, remove this BN layer.Default: 0
    '''

    def __init__(self, num_features, ratio=0.5, detach=True, mode=0,
                momentum=0.1, eps=1e-5, **kargs):
        super().__init__(num_features, eps=eps, momentum=momentum, **kargs)

        assert ratio>=0 and ratio <=1
        self.ratio = ratio
        self.detach = detach
        self.mode = mode
        print_log(f'Mix BN: mode= {mode}, ratio= {ratio}, detach= {detach}', logger='mmseg')

        # shape = (1, num_features, 1, 1)
        self.running_mean_dst = torch.zeros(num_features).cuda()
        self.running_var_dst = torch.ones(num_features).cuda()

    def forward(self, input:Optional[Tensor], domain):
        '''
        Args:
            domain (list): list indicates source domain and target domain, 0
                indicates source, 1 indicates target            
        '''

        if not torch.is_grad_enabled():
            if self.mode == 0:
                return super().forward(input)
            elif self.mode == 1:
                return input
            elif self.mode == 2:
                return super().forward(input)
            else:
                raise NotImplementedError
        else:
            src_input, dst_input = Semi.split_domins_data(input, domain=domain)
            
            # dst params
            dst_output, _, _, mean_dst, var_dst = mix_bn(
                dst_input, self.weight, self.bias, self.running_mean_dst, self.running_var_dst,
                self.eps, self.momentum, ratio=1, mean=0, var=0)

            if self.detach:
                mean_dst = mean_dst.detach()
                var_dst = var_dst.detach()

            # src params
            if self.mode == 0:
                ''' replace source mean and var with targe mean and var'''
                src_ouput, self.running_mean, self.running_var, _, _ = mix_bn(src_input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.momentum, self.ratio, mean_dst, var_dst)
            elif self.mode == 1:
                ''' replace source mean and var with targe bias and weight, without leanable weight and bias'''
                mean_src = src_input.mean(dim=(0, 2, 3), keepdim=True)
                var_src = ((src_input - mean_src)**2).mean(dim=(0, 2, 3), keepdim=True)
                bias = self.ratio * mean_src + (1-self.ratio)* mean_dst
                weight = self.ratio * var_src + (1-self.ratio) * var_dst
                bias = bias.squeeze()
                weight = bias.squeeze()
                src_ouput = F.batch_norm(src_input, self.running_mean, self.running_var, weight, bias, True, self.momentum, self.eps)
            elif self.mode == 2:
                ''' replace source mean and var with targe bias and weight, with leanable weight and bias '''
                bias = self.ratio * self.bias + (1-self.ratio)* mean_dst.squeeze()
                weight = self.ratio * self.weight + (1-self.ratio) * var_dst.squeeze()
                src_ouput = F.batch_norm(src_input, self.running_mean, self.running_var, weight, bias, True, self.momentum, self.eps)
            else:
                raise NotImplementedError
            
            output = Semi.merge_domains_data((src_ouput, dst_output), domain=domain)
            
            # output = torch.empty_like(input)
            # output[dst_idx, ...] = dst_output
            # output[src_idx, ...] = src_ouput
            
            return output


def mix_bn(X, weight, bias, running_mean, running_var, eps, momentum, ratio,
            mean=0, var=0):
    ''' BatchNorm implementation from dive into deep learning
    '''

    if not torch.is_grad_enabled():
        # deprecated !!!
        weight = weight.reshape(1, weight.shape[0], 1, 1)
        bias = bias.reshape(1, bias.shape[0], 1, 1)
        X_hat = (X - running_mean) / torch.sqrt(running_var + eps)
    else:
        local_mean = X.mean(dim=(0, 2, 3), keepdim=True)
        local_var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)

        mean = ratio * local_mean + (1-ratio)*mean
        var = ratio * local_var + (1-ratio)*var

        X_hat = (X - mean) / torch.sqrt(var + eps)

        # Update the mean and variance using running average
        running_mean = momentum * mean.squeeze() \
                        + (1.0 - momentum) * running_mean
        running_var = momentum * var.squeeze() \
                        + (1.0 - momentum) * running_var

    # reshape for broadcasting
    weight = weight.reshape(1, weight.shape[0], 1, 1)
    bias = bias.reshape(1, bias.shape[0], 1, 1)
    Y = weight * X_hat + bias           # Scale and shift

    return Y, running_mean.data, running_var.data, mean, var

