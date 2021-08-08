'''
Author: Shuailin Chen
Created Date: 2021-07-11
Last Modified: 2021-07-11
	content: 
'''

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/sar_building_mix.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

# for schedule
lr_config = dict(policy='poly', power=0.9, min_lr=0.001, by_epoch=False)
total_iters = 1600
checkpoint_config = dict(by_epoch=False, interval=100)
evaluation = dict(interval=100, metric='mIoU')

gpu_ids = range(0, 1)