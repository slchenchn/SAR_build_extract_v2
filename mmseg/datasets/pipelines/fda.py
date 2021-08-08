'''
Author: Shuailin Chen
Created Date: 2021-07-10
Last Modified: 2021-07-12
	content: adapted from https://github.com/NoelShin/PixelPick/blob/master/utils/utils.py
'''
import os.path as osp
import mmcv
import torch
import numpy as np
from numpy import random
from glob import glob
import cv2
from mylib import image_utils as iu

from ..builder import PIPELINES

@PIPELINES.register_module()
class FourierDomainAdaption(object):
    ''' Fourier domain adaption, adapted from paper "FDA: Fourier Domain Adaptation for Semantic Segmentation"
    
    Args:
        LB (float): size of the low frequency window to be replaced, must 
            between 0 and 1. Default: 0.01
    '''
    def __init__(self, dst_img_dir, LB=0.01):
        assert LB>=0 and LB<=1, f'LB must in [0, 1], but got {LB}'
        self.LB = LB
        self.dst_img_dir = dst_img_dir
        self.dst_imgs = list(mmcv.scandir(dst_img_dir, suffix='.png', recursive=True))
    
    def __len__(self):
        """Total number of images in dst_img_dir."""
        return len(self.dst_imgs)

    def __call__(self, results):
        
        # load image
        idx = np.random.randint(0, len(self))
        dst_img = cv2.imread(osp.join(self.dst_img_dir, self.dst_imgs[idx]))
        src_img = results['img']
        src_img = src_img.transpose((2, 0, 1))
        dst_img = dst_img.transpose((2, 0, 1))

        new_src_img = FDA_source_to_target_np(src_img, dst_img, L=self.LB)
        new_src_img = np.clip(new_src_img, a_max=255, a_min=0)
        new_src_img = new_src_img.astype(np.uint8)
        new_src_img = new_src_img.transpose((1, 2, 0))
        results['img'] = new_src_img
        results['FDA'] = True

        # src_img = src_img.transpose((1, 2, 0))
        # dst_img = dst_img.transpose((1, 2, 0))
        # iu.save_image_by_cv2(src_img, r'./tmp/src.jpg', is_bgr=True, if_norm=False)
        # iu.save_image_by_cv2(dst_img, r'./tmp/dst.jpg', is_bgr=True, if_norm=False)
        # iu.save_image_by_cv2(new_src_img, r'./tmp/newsrc.jpg', is_bgr=True, if_norm=False)
        # print(results['filename'])
        
        return results






def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha


def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src


def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    # why choose a square spectrum, not a shape like the original image?
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src


def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude for torch tensor in shape of [batch, channel, height, weight]
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False ) 
    fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    # recompose fft of source
    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )

    return src_in_trg


def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude for numpy ndarray in shape of [channel, height, weight]
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg

