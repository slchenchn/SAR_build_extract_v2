'''
Author: Shuailin Chen
Created Date: 2021-07-12
Last Modified: 2021-08-10
	content: 
''' 
from torch import Tensor
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.ops import resize

from .decode_head import BaseDecodeHead
from ..segmentors import Semi
from ..losses import accuracy


class BaseDecodeHeadMixBN(BaseDecodeHead):
    """
    """

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, domain):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs, domain=domain)
        
        seg_logits_src, _ = Semi.split_domins_data(seg_logits, domain=domain)
        gt_semantic_seg_src, _ = Semi.split_domins_data(gt_semantic_seg, domain=domain)
        
        losses = self.losses(seg_logits_src, gt_semantic_seg_src)
        return losses


