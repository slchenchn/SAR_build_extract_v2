'''
Author: Shuailin Chen
Created Date: 2021-09-02
Last Modified: 2021-09-04
	content: 
'''
from .cgnet import CGNet
from .fast_scnn import FastSCNN
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
from .swin import SwinTransformer
from .unet import UNet
from .vit import VisionTransformer

from .resnet_mixbn import ResNetMixBN, ResNetV1cMixBN, ResNetV1dMixBN
from .resnet_multich import ResNetMulti,ResNetV1cMulti,ResNetV1dMulti

# __all__ = [
#     'ResNet', 'ResNetV1c', 'ResNetV1d', 'ResNeXt', 'HRNet', 'FastSCNN',
#     'ResNeSt', 'MobileNetV2', 'UNet', 'CGNet', 'MobileNetV3',
#     'VisionTransformer', 'SwinTransformer'
# ]
