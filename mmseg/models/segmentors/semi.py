'''
Author: Shuailin Chen
Created Date: 2021-08-04
Last Modified: 2021-08-04
	content: 
'''

from torch import nn
import torch
import numpy as np

from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import initialize
from mmcv.runner import auto_fp16

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class Semi(EncoderDecoder):
    ''' Mix BatchNorm parameters for domain adaptation
    '''

    def __init__(self, detach=False, **kargs):
        self.detach = detach
        super().__init__(**kargs)
        
    @staticmethod
    def get_domains_slice(domain):

        dst_idx = np.argwhere(domain).flatten()
        src_idx = np.argwhere(np.logical_not(domain)).flatten()

        return src_idx, dst_idx

    @staticmethod
    def check_domain(data, domain):
        if domain is None:
            if isinstance(data, list):
                domain = np.zeros(len(data))
            elif isinstance(data, torch.Tensor):
                domain = np.zeros(data.shape[0])
            else:
                raise NotImplementedError
        return domain

    @staticmethod
    def merge_domains_data(data, domain):
        assert isinstance(data, (tuple, list))
        assert data[0].shape[0] + data[1].shape[0] == domain.size
        assert data[0].shape[1:] == data[1].shape[1:]

        new_data = torch.empty(size=(domain.size, *data[0].shape[1:]), device=data[0].device, dtype=data[0].dtype)
        src_idx, dst_idx = Semi.get_domains_slice(domain=domain)
        new_data[src_idx, ...] = data[0]
        new_data[dst_idx, ...] = data[1]

        return new_data

    @staticmethod
    def split_domins_data(data, domain):
        assert domain.size == data.shape[0]

        src_idx, dst_idx = Semi.get_domains_slice(domain=domain)
        src_data = data[src_idx, ...]
        dst_data = data[dst_idx, ...]

        return src_data, dst_data
        
    def extract_feat(self, img, domain):
        """Extract features from images."""
        domain = Semi.check_domain(data=img, domain=domain)
        x = self.backbone(img, domain=domain)
        if self.with_neck:
            x = self.neck(x, domain=domain)
        return x

    @auto_fp16(apply_to=('img', ))
    def forward_train(self, img, img_metas, gt_semantic_seg):
        ''' Main body of network forward function 
        '''

        # extract domain index
        domain = np.array([meta['domain'] for meta in img_metas], dtype=np.int64)        
        x = self.extract_feat(img, domain=domain)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg, domain=domain)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, domain=domain)
            losses.update(loss_aux)

        return losses

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            # pass
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)


    # def init_weights(self):
    #     """Initialize the weights."""

    #     if not self._is_init:
    #         if self.init_cfg:
    #             initialize(self, self.init_cfg)
    #             if isinstance(self.init_cfg, (dict, ConfigDict)):
    #                 # Avoid the parameters of the pre-training model
    #                 # being overwritten by the init_weights
    #                 # of the children.
    #                 if self.init_cfg['type'] == 'Pretrained':
    #                     return

    #         for m in self.children():
    #             if hasattr(m, 'init_weights'):
    #                 m.init_weights()
    #         self._is_init = True
    #     else:
    #         warnings.warn(f'init_weights of {self.__class__.__name__} has '
    #                     f'been called more than once.')
