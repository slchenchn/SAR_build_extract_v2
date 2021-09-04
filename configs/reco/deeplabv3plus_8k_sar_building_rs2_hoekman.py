'''
Author: Shuailin Chen
Created Date: 2021-09-04
Last Modified: 2021-09-04
	content: 
'''

_base_=[
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/sar_building_rs2_hoekman.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

model = dict(
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        # type='ResNetV1cMulti',
        in_channels = 9,
    )

)


runner = dict(type='IterBasedRunner', max_iters=8000)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.001, by_epoch=False)
checkpoint_config = dict(by_epoch=False, interval=300000)
evaluation = dict(interval=800, metric='mIoU')