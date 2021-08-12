'''
Author: Shuailin Chen
Created Date: 2021-08-04
Last Modified: 2021-08-12
	content: 
'''

from torch import nn
import torch
import numpy as np

from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import initialize
from mmcv.runner import auto_fp16
import mmcv.parallel.collate
import mmcv.runner.iter_based_runner

from mmseg.core import add_prefix

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class Semi(EncoderDecoder):
    ''' Encoder decoder framwork for domain adaptation or semi supervision
    '''
        
    @staticmethod
    def get_domains_slice(domain):
        ''' Get domains' slice, as the input, the domain indicates which domian every sample belongs to, like [0, 1, 0, 1], this func convert it into separate domain index, like src_idx=[0, 2], dst_idx=[1, 3], for the convenience of separate domain data

        NOTE: this is just a realization of two domains
        '''

        # flatten into 1D array
        dst_idx = np.argwhere(domain).flatten()
        src_idx = np.argwhere(np.logical_not(domain)).flatten()

        return src_idx, dst_idx

    @staticmethod
    def check_domain(data, domain):
        ''' Check whether the size of domain matches the size of input data, or if the domain is None, set the domain to all 0s. Almostly, only the extract_feat(), decode_head_forward_train need check domain, not a single conv layer
        
        Args:
            data (list | Tensor): input data    
        '''

        if isinstance(data, list):
            data_len = len(data)
        elif isinstance(data, torch.Tensor):
            data_len = data.shape[0]
        else:
            raise NotImplementedError(f'unsupported data type: {type(data)}')

        if domain is None:
            domain = np.zeros(data_len)
        else:
            assert domain.size == data_len

        # if domain is None:
        #     if isinstance(data, list):
        #         domain = np.zeros(len(data))
        #     elif isinstance(data, torch.Tensor):
        #         domain = np.zeros(data.shape[0])
        #     else:
        #         raise NotImplementedError
        # else:
        #     if isinstance(data, list):
        #         assert domain.size == len(data)
        #     elif isinstance(data, torch.Tensor):
        #         assert domain.size == data.shape[0]
        #     else:
        #         raise NotImplementedError

        return domain

    @staticmethod
    def merge_domains_data(data, domain):
        ''' Merge data of two domains accroding to domain slice

        Args:
            domain (list[int]): use an unique integer to represent a domain, 
                e.g., [0, 1, 0, 1]
            data (tuple[Tensor]): data[0] correspond to 0 in domain, while data
                [1] correspond 1 in domain
        '''

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
        ''' split data into two domains accroding to domain slice

        Args:
            domain (list[int]): use an unique integer to represent a domain, 
                e.g., [0, 1, 0, 1]
            data (Tensor): input Tensor contrains data from both domains
            
        Returns:
            src_data, dst_data (Tensor): data from source/destination domain
        '''

        assert domain.size == data.shape[0]

        src_idx, dst_idx = Semi.get_domains_slice(domain=domain)
        src_data = data[src_idx, ...]
        dst_data = data[dst_idx, ...]

        return src_data, dst_data
        
    def extract_feat(self, img, domain=None):
        """ Extract features by backbone and neck, add keyword: domain for seperated domain training purpose

        Args:
            domain (list[int]): use an unique integer to represent a domain, 
                e.g., [0, 1, 0, 1]
        """

        # for traing, the domain keyword is given; while for eval, all data use the same model, so distinguish differenct domain is unnecessry
        domain = Semi.check_domain(data=img, domain=domain)

        x = self.backbone(img, domain=domain)
        if self.with_neck:
            x = self.neck(x, domain=domain)
        return x

    @auto_fp16(apply_to=('img', ))
    def forward_train(self, img, img_metas, gt_semantic_seg):
        ''' Main body of network forward function, contains backbone, neck, decode_head
        '''

        # extract domain index through img_mertas
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

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg, domain=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg, domain=domain)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg, domain=domain)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, domain):
        """Run forward function and calculate loss for decode head in
        training."""
        
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg, domain=domain)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
