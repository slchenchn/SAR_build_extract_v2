#
the v1 version's environments is direct copied from ghw's, which will cause problems because the mmsegmentation site-package is installed from code, and can't be edited by me.

#
`FDA`: the training process is unstable, maybe this method is not suitable for PolSAR building extraction

# BN mixup
failed, see onenote for detail

## modified part of MMCV：

	1. ~/anaconda3/envs/sar_build_extract_v2/lib/python3.7/site-packages/mmcv/parallel/collate.py

    2. ~/anaconda3/envs/sar_build_extract_v2/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py
