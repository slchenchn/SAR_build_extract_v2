export CUDA_VISIBLE_DEVICES=0
python inference.py ./sar_building/fcn_hr48.py	./work_dirs/fcn_hr48_512x512_80k_ade20k/iter_1600.pth	/data2/ghw/sar_building/images/test	/data2/ghw/sar_segm_inference/mixed2/hrnet
python inference.py ./sar_building/pspnet_r50-d8.py	./work_dirs/pspnet_r50-d8_512x512_80k_ade20k/iter_1400.pth	/data2/ghw/sar_building/images/test	/data2/ghw/sar_segm_inference/mixed2/pspnet
python inference.py ./sar_building/ccnet_r50.py	./work_dirs/ccnet_r50-d8_512x512_80k_ade20k/iter_1600.pth	/data2/ghw/sar_building/images/test	/data2/ghw/sar_segm_inference/mixed2/ccnet
python inference.py ./sar_building/gcnet_r50.py	./work_dirs/gcnet_r50-d8_512x512_80k_ade20k/iter_1300.pth	/data2/ghw/sar_building/images/test	/data2/ghw/sar_segm_inference/mixed2/gcnet
python inference.py ./sar_building/psanet_r50.py	./work_dirs/psanet_r50-d8_512x512_80k_ade20k/iter_1600.pth	/data2/ghw/sar_building/images/test	/data2/ghw/sar_segm_inference/mixed2/psanet
python inference.py ./sar_building/upernet_r50.py	./work_dirs/upernet_r50_512x512_80k_ade20k/iter_1600.pth	/data2/ghw/sar_building/images/test	/data2/ghw/sar_segm_inference/mixed2/upernet
