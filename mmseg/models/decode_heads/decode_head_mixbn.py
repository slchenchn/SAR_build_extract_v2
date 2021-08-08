'''
Author: Shuailin Chen
Created Date: 2021-07-12
Last Modified: 2021-07-14
	content: 
''' 

from mmcv.runner import BaseModule, auto_fp16, force_fp32

from .decode_head import BaseDecodeHead
from ..segmentors import Semi


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
        
        seg_logits_src = Semi.split_domins_data(seg_logits, domain=domain)
        gt_semantic_seg = Semi.split_domins_data(gt_semantic_seg, domain=domain)
        
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss.
        
        Args:
            seg_logit (list | Tensor): Tensor for original operation
                chain, list for new operation chain
        """

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None

        loss = dict()
        if isinstance(seg_logit, Tensor):
            seg_logit = resize(
                input=seg_logit,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            seg_label = seg_label.squeeze(1)

            loss['acc_seg'] = accuracy(seg_logit, seg_label)

        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)

        return loss
