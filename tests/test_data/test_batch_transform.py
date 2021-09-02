'''
Author: Shuailin Chen
Created Date: 2021-09-02
Last Modified: 2021-09-02
	content: Undone
'''
import pytest
import os.path as osp
import mmcv
from mmcv.utils import build_from_cfg

from mmseg.datasets.builder import PIPELINES


def test_batch_normalize():
    # test assertion if img_scale is a list
    img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    
    ori_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Normalize', **img_norm_cfg),
    ]
        transform = dict(type='BatchNormalize', **img_norm_cfg)
        build_from_cfg(transform, PIPELINES)

    norm_module = build_from_cfg(transform, PIPELINES)

    results = dict()
    # (288, 512, 3)
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] == (750, 1333, 3)

    # test keep_ratio=False
    transform = dict(
        type='Resize',
        img_scale=(1280, 800),
        multiscale_mode='value',
        keep_ratio=False)
    resize_module = build_from_cfg(transform, PIPELINES)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] == (800, 1280, 3)

    # test multiscale_mode='range'
    transform = dict(
        type='Resize',
        img_scale=[(1333, 400), (1333, 1200)],
        multiscale_mode='range',
        keep_ratio=True)
    resize_module = build_from_cfg(transform, PIPELINES)
    resized_results = resize_module(results.copy())
    assert max(resized_results['img_shape'][:2]) <= 1333
    assert min(resized_results['img_shape'][:2]) >= 400
    assert min(resized_results['img_shape'][:2]) <= 1200

    # test multiscale_mode='value'
    transform = dict(
        type='Resize',
        img_scale=[(1333, 800), (1333, 400)],
        multiscale_mode='value',
        keep_ratio=True)
    resize_module = build_from_cfg(transform, PIPELINES)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] in [(750, 1333, 3), (400, 711, 3)]

    # test multiscale_mode='range'
    transform = dict(
        type='Resize',
        img_scale=(1333, 800),
        ratio_range=(0.9, 1.1),
        keep_ratio=True)
    resize_module = build_from_cfg(transform, PIPELINES)
    resized_results = resize_module(results.copy())
    assert max(resized_results['img_shape'][:2]) <= 1333 * 1.1

    # test img_scale=None and ratio_range is tuple.
    # img shape: (288, 512, 3)
    transform = dict(
        type='Resize', img_scale=None, ratio_range=(0.5, 2.0), keep_ratio=True)
    resize_module = build_from_cfg(transform, PIPELINES)
    resized_results = resize_module(results.copy())
    assert int(288 * 0.5) <= resized_results['img_shape'][0] <= 288 * 2.0
    assert int(512 * 0.5) <= resized_results['img_shape'][1] <= 512 * 2.0