'''
Author: Shuailin Chen
Created Date: 2021-07-11
Last Modified: 2021-08-28
	content: 
'''

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/sar_building_gf3_to_rs2.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

# change num_classes
model = dict(
    decode_head=dict(
        num_classes=2,
    ),
    auxiliary_head=dict(
        num_classes=2,
    ),
)

# for schedule
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.005)
lr_config = dict(policy='poly', power=0.9, min_lr=0.001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=800)
checkpoint_config = dict(by_epoch=False, interval=100)
evaluation = dict(interval=100, metric='mIoU')

gpu_ids = range(0, 1)