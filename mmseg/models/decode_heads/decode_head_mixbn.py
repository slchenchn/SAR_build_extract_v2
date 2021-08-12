'''
Author: Shuailin Chen
Created Date: 2021-07-12
Last Modified: 2021-08-12
	content: 
''' 

from .decode_head import BaseDecodeHead
from ..segmentors import Semi


class BaseDecodeHeadMixBN(BaseDecodeHead):
    """ Base class for BaseDecodeHead for domain adaptation, the difference between original and this version is the forward() func
    """

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, domain):
        seg_logits = self.forward(inputs, domain=domain)
        
        seg_logits_src, _ = Semi.split_domins_data(seg_logits, domain=domain)
        gt_semantic_seg_src, _ = Semi.split_domins_data(gt_semantic_seg, domain=domain)
        
        losses = self.losses(seg_logits_src, gt_semantic_seg_src)
        return losses


