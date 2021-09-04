'''
Author: Shuailin Chen
Created Date: 2021-08-28
Last Modified: 2021-09-03
	content: 
'''

_base_ = [
    './deeplabv3plus_reco_4k_s_s_sar_build.py'
]


runner = dict(type='SemiIterBasedRunner', max_iters=8000)

evaluation = dict(interval=800, metric='mIoU')

