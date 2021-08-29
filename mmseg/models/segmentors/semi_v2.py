'''
Author: Shuailin Chen
Created Date: 2021-08-04
Last Modified: 2021-08-28
	content: 
'''

from torch import nn
import torch
import numpy as np
from abc import ABCMeta, abstractmethod

from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import initialize
from mmcv.runner import auto_fp16

from mmseg.core import add_prefix

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class SemiV2(EncoderDecoder):
    ''' Second version of encoder decoder framwork for semi supervision, this version should be used with SemiIterBasedRunner
    '''
    
    def train_step(self, data_batch:dict, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloaders.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        # parse the data_batch in `forward` func
        losses = self(data_batch)

        # TODO: need change the loss parsing progress
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    @auto_fp16(apply_to=('img', ))
    def forward(self, data_batch, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        
        assert isinstance(data_batch, dict)
        
        if return_loss:
            assert len(data_batch)==1
        else:
            assert len(data_batch)>1

        if return_loss:
            return self.forward_train(**data_batch, **kwargs)
        else:
            return self.forward_test(*data_batch.values(), **kwargs)

    @abstractmethod
    def forward_train(self, labeled:dict, unlabeled:dict, **kargs):
        """Forward function for training.

        Args:
            labeled & unlabeled(dict): labeled and unlabeled data batch

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        pass