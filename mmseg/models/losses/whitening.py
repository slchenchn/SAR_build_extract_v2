'''
Author: Shuailin Chen
Created Date: 2021-07-14
Last Modified: 2021-07-15
	content: feature whitening loss, mostly adapted from https://github.com/shachoi/RobustNet
'''

import torch
import torch.nn as nn

from ..builder import LOSSES


@LOSSES.register_module()
class RelaxedInstanceWhiteningLoss(nn.Module):
    ''' Relaxed instance whitening loss

    Args:
        stage_idx (list): indicates to which stage apply this loss
        relax_denom (float): relax denominator value
    '''
    
    def __init__(self, relax_denom=64, loss_weight=1.0):
        super().__init__()
        self.relax_denom = relax_denom
        self.loss_weight = loss_weight

    def forward(self, seg_logit, seg_label, weight, ignore_index):
        '''
        Args:
            seg_logit (Tensor): feature maps, use this name to acommodate mmsegmentation framwork
            seg_label: NOT used here, just to acommodate mmseg
            weight: NOT used here
            ignore_index: NOT used here
        '''
        loss = 0
        for f_map in seg_logit:
            loss += instance_whitening_loss(f_map, relax_denom=self.relax_denom)
            
        loss = loss / len(seg_logit)
        return self.loss_weight * loss


def instance_whitening_loss(f_map, relax_denom, eye=None, mask_matrix=None, ):
    ''' Instance whitening loss
    
    Args:
        f_map (Tensor): torch feature map
        eye (Tensor): eye matrix
        mask_matrix (Tensor): mask to perform whitening loss
        relax_denom (float): margin to calculate loss

    Returns:
        loss
    '''

    channels = f_map.shape[1]
    if mask_matrix is None:
        mask_matrix = torch.ones(channels, channels).triu(diagonal=1).cuda()
 
    num_remove_cov = torch.sum(mask_matrix)
    margin = num_remove_cov / relax_denom

    f_cor, B = get_covariance_matrix(f_map, eye=eye)
    f_cor_masked = f_cor * mask_matrix

    off_diag_sum = torch.sum(torch.abs(f_cor_masked), dim=(1,2), keepdim=True) - margin # B X 1 X 1
    loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0) # B X 1 X 1
    loss = torch.sum(loss) / B

    return loss


def get_covariance_matrix(f_map, eye=None):
    ''' Get covariance matrix of a torch feature map
    
    Args:
        f_map (Tensor): torch feature map
        eye (Tensor): eye matrix
    
    Returns:
        f_cor (Tensor): covariance matrix of feature map
        B (int): batch size
    '''
    eps = 1e-5
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)  # C X C / HW

    return f_cor, B