'''
Author: Shuailin Chen
Created Date: 2021-08-28
Last Modified: 2021-09-02
	content: 
'''
from copy import deepcopy
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from torchvision.transforms.functional import normalize
import mylib.image_utils as iu

from mmseg.ops import resize
from mmseg.core import add_prefix
from mmseg.datasets.pipelines import Compose
from ..builder import SEGMENTORS
from .semi_v2 import SemiV2


@SEGMENTORS.register_module()
class ReCo(SemiV2):
    ''' my implementation of regional contrast algorithm for semi-supervised semantic segmentation

    Args:
        momentum (float): momentum to update the mean teacher model, must
            between [0, 1]. Default: 0.99
        strong_thres (float): strong threshold to filter the difficult
            samples, must between [0, 1]. Default: 0.97
        weak_thres (float): weak threshold to filter the unsure
            samples, must between [0, 1]. Default: 0.7
        tmperature (float): tmperature in the contrastive loss. Default: 0.5
        num_queries (int): number of queries in the contrastive loss.
            Default: 256
        num_negatives (int): number of negative samples in the contrastive
            loss. Default: 512
        apply_reco (bool): whether to apply regional contrast loss. 
            Default: True
        apply_pseudo_loss (bool): whether to apply pseudo labeling loss. 
            Default: True
        unlabeled_aug (list[dict]): Processing pipeline for unlabled data
    '''

    def __init__(self, 
                momentum = 0.99,
                strong_thres = 0.97,
                weak_thres = 0.7,
                tmperature = 0.5,
                num_queries = 256,
                num_negatives = 512,
                apply_reco=True,
                apply_pseudo_loss=True,
                unlabeled_aug=None,
                **kargs):
        super().__init__(**kargs)
        self.momentum = momentum
        self.strong_thres = strong_thres
        self.weak_thres = weak_thres
        self.tmperature = tmperature
        self.num_queries = num_queries
        self.num_negatives = num_negatives
        self.apply_reco = apply_reco
        self.apply_pseudo_loss = apply_pseudo_loss
        self.unlabeled_aug = {k:Compose(v) for k, v in unlabeled_aug.items()}

        # for updating EMA model
        self.step = 0

    def init_weights(self):
        super().init_weights()

        # init EMA model as the student
        self.backbone_ema = deepcopy(self.backbone)
        self.decode_head_ema = deepcopy(self.decode_head)
        if self.with_neck:
            self.neck_ema = deepcopy(self.neck)

    @staticmethod
    def ema_update(ema, model, decay):
        ''' update the EMA model with decay params specified '''
        for ema_param, param in zip(ema.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data

    def ema_update_whole(self):
        ''' Update the EMA (mean teacher) model '''
        decay = min(1 - 1 / (self.step + 1), self.momentum)
        self.step += 1

        ReCo.ema_update(self.backbone_ema, self.backbone, decay)
        ReCo.ema_update(self.decode_head_ema, self.decode_head, decay)
        if self.with_neck:
            ReCo.ema_update(self.neck_ema, self.neck, decay)

    def ema_forward(self, img):
        ''' Forward function of EMA model, not including the loss calculation '''
        x = self.backbone_ema(img)
        if self.with_neck:
            x = self.neck_ema(x)
        preds, _ = self.decode_head_ema.forward(x)
        return preds

    def main_forward(self, img, img_metas=None, gt_semantic_seg=None):
        ''' Forward function of the main model, also calculate loss if GT is given
        
        Returns:
            preds (Tensor): prediction logits
            reps (Tensor): representation tensor, for purpose of regional contrast
        '''
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        preds, reps = self.decode_head.forward(x)

        # calculate loss if GT is given
        loss = dict()
        if (img_metas is not None) and (gt_semantic_seg is not None):
            decode_losses = self.decode_head.losses(preds, gt_semantic_seg)
            decode_losses = add_prefix(decode_losses, 'decode')
            
            if self.with_auxiliary_head:
                aux_loss = self._auxiliary_head_forward_train(
                    x, img_metas, gt_semantic_seg)

            loss.update(decode_losses)
            loss.update(aux_loss)

        return preds, reps, loss

    def forward_train(self, labeled: dict, unlabeled: dict, **kargs):

        # generate strong and weak augmented unlabeled data
        # NOTE: when use one variable for multiple times, BE CAREFUL to the inplace opterations !!!
        unlabeled = {k: v(deepcopy(unlabeled))
                        for k, v in self.unlabeled_aug.items()}

        for name, batch in unlabeled.items():
            batch['img'] = batch['img'].to(labeled['img'].device)

        # generate pseudo labels for weak augmented unlabeled data
        self.ema_update_whole()
        with torch.no_grad():
            ema_pred = self.ema_forward(unlabeled['weak']['img'])
            ema_pred = resize(input=ema_pred,
                                    size=labeled['img'].shape[2:],
                                    mode='bilinear',
                                    align_corners=self.align_corners)
                                    
            pseudo_probs, pseudo_labels = torch.max(torch.softmax(ema_pred, dim=1), dim=1)

        # supervised loss
        preds_l, reps_l, sup_loss = self.main_forward(labeled['img'], 
                                                    labeled['img_metas'], labeled['gt_semantic_seg'])
        preds_u, reps_u, _ = self.main_forward(unlabeled['strong']['img'])

        rep_all = torch.cat((reps_l, reps_u))
        pred_all = torch.cat((preds_l, preds_u))

        # pseudo label loss
        if self.apply_pseudo_loss:
            unsup_loss = self.compute_unsupervised_loss(preds_u, pseudo_labels,
                                                    pseudo_probs)
        else:
            unsup_loss = torch.tensor(0.0, device=labeled['img'].device)

        # ReCo loss
        if self.apply_reco:
            with torch.no_grad():
                # mask
                pseudo_mask = pseudo_probs.ge(self.weak_thres)
                mask_all = torch.cat((labeled['gt_semantic_seg']>=0,
                                    pseudo_mask.unsqueeze(1)))
                mask_all = resize(mask_all.float(), size=pred_all.shape[2:])

                # label
                one_hot_label = self.label_onehot(labeled['gt_semantic_seg'])
                one_hot_pseudo_label = self.label_onehot(
                                                    pseudo_labels.unsqueeze(1))
                label_all = torch.cat((one_hot_label, one_hot_pseudo_label))
                label_all = resize(label_all.float(), size=pred_all.shape[2:])

                # predicted probability
                prob_l = torch.softmax(preds_l, dim=1)
                prob_u = torch.softmax(preds_u, dim=1)
                prob_all = torch.cat((prob_l, prob_u))

            reco_loss = self.compute_reco_loss(rep_all, label_all, mask_all, prob_all)
        else:
            reco_loss = torch.tensor(0.0, device=labeled['img'].device)

        loss = dict()
        loss.update(sup_loss)
        loss.update({'unsup.loss': unsup_loss, 'reco.loss': reco_loss})

        return loss

    def compute_unsupervised_loss(self, predict, target, probs):
        ''' Compute pseudo labeling loss

        Args:
            predict (Tensor): model prediction logits
            target (Tensor): pseudo labels
            probs (Tensor): probilities corresponding pseudo labels
        '''

        # resize first
        predict = resize(
                input=predict,
                size=target.shape[1:],
                mode='bilinear',
                align_corners=self.align_corners)

        batch_size = predict.shape[0]
        valid_mask = (target >= 0).float()   # only count valid pixels

        # 对每个样本进行加权
        weighting = probs.view(batch_size, -1).ge(self.strong_thres).sum(-1) / valid_mask.view(batch_size, -1).sum(-1)
        loss = F.cross_entropy(predict, target, reduction='none', ignore_index=-1)
        # NOTE: there are unlabeled pixels in cityscapes, so here need to mask out them
        weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None] * loss, loss > 0))
        return weighted_loss

    def label_onehot(self, inputs):
        ''' Convert indexed label to one-hot vector '''
        batch_size, _, im_h, im_w = inputs.shape
        num_classes = self.decode_head.num_classes
        # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
        # we will still mask out those invalid values in valid mask
        inputs = torch.relu(inputs)
        outputs = torch.zeros([batch_size, num_classes, im_h, im_w]).to(inputs.device)
        return outputs.scatter_(1, inputs, 1.0)

    def compute_reco_loss(self, rep, label, mask, prob):
        ''' Compute regional contrast loss, contrast pixel embeddings with class meann embeddings actually

        Args:
            rep (Tensor): representation of all samples
            label (Tensor): one-hot labels of all samples
            mask (Tensor): all elements of labeled images, and elements with
                confidence greater than weak threshold in unlabeled images
            prob (Tensor): probabilities of all samples
        '''
        _, num_feat, im_w_, im_h = rep.shape
        device = rep.device

        # compute valid binary mask for each pixel, shape: BxCxHxW
        valid_pixel = label * mask

        # permute representation for indexing: B x H x W x C
        rep = rep.permute(0, 2, 3, 1)

        # compute prototype (class mean representation) for each class across all valid pixels
        feat_all_list = []
        feat_hard_list = []
        seg_num_list = []
        proto_list = []
        for i in range(self.num_classes):

            #select binary mask for i-th class
            valid_pixel_seg = valid_pixel[:, i, ...]
            if valid_pixel_seg.sum() == 0:  
                ''' not all classes would be available in a mini-batch '''
                continue

            prob_seg = prob[:, i, :, :]

            # select hard pixels (confidence < strong threshold), besides, for unlabeled pixels, its confidence should also be greater than the weak threshold
            rep_mask_hard = (prob_seg < self.strong_thres) * valid_pixel_seg.bool()

            # generate prototypes' embeddings
            proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
            feat_all_list.append(rep[valid_pixel_seg.bool()])
            feat_hard_list.append(rep[rep_mask_hard])
            # number of valid pixels of the ith class
            seg_num_list.append(int(valid_pixel_seg.sum().item()))  

        # compute regional contrastive loss
        if len(seg_num_list) <= 1:  
            ''' in some rare cases, a small mini-batch might only contain 1 or no semantic class '''
            return torch.tensor(0.0, device=device)
        else:
            reco_loss = torch.tensor(0.0, device=device)
            seg_proto = torch.cat(proto_list)   #prototype
            valid_seg = len(seg_num_list)
            seg_len = torch.arange(valid_seg)

            for i in range(valid_seg):
                if len(feat_hard_list[i]) > 0:
                    # sample fixed number of hard queries, even the actual number of queries small than required
                    seg_hard_idx = torch.randint(len(feat_hard_list[i]), size=(self.num_queries,))
                    anchor_feat_hard = feat_hard_list[i][seg_hard_idx]
                    anchor_feat = anchor_feat_hard
                else:  
                    ''' in some rare cases, all queries in the current query class are easy '''
                    continue

                # negative key sampling (with no gradients)
                with torch.no_grad():
                    # generate index mask for the current query class,
                    # e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
                    seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

                    # compute similarity for each segment prototype (semantic class relation graph) for the following negative pixel sampling
                    proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1)
                    proto_prob = torch.softmax(proto_sim / self.tmperature,
                                                dim=0)

                    # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                    negative_dist = Categorical(probs=proto_prob)
                    samp_class = negative_dist.sample(
                        sample_shape=[self.num_queries, self.num_negatives])
                    samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

                    # sample negative indices from each negative class
                    negative_num_list = seg_num_list[i+1:] + seg_num_list[:i]
                    negative_index = negative_index_sampler(samp_num,
                                                            negative_num_list)

                    # index negative keys (from other classes)
                    negative_feat_all = torch.cat(feat_all_list[i+1:] + feat_all_list[:i])
                    negative_feat = negative_feat_all[negative_index].reshape(self.num_queries, self.num_negatives, num_feat) 

                    # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                    positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(self.num_queries, 1, 1)
                    all_feat = torch.cat((positive_feat, negative_feat), dim=1)

                seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
                reco_loss += F.cross_entropy(seg_logits / self.tmperature, torch.zeros(self.num_queries).long().to(device))
            return reco_loss / valid_seg

    def encode_decode(self, img, img_metas):
        """ Overload the original func."""
        x = self.extract_feat(img)
        out, _ = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[:j]),
                                                high=sum(seg_num_list[:j+1]),
                                                size=int(samp_num[i, j])).tolist()
    return negative_index