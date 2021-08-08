export CUDA_VISIBLE_DEVICES=0

python inference.py ./sar_building/deeplabv3p.py	./work_dirs/deeplabv3plus_r50-d8_512x512_80k_ade20k/iter_1500.pth	/data2/ghw/sar_building/images/test	/data2/ghw/sar_segm_inference/mixed/deeplabv3p

python inference.py ./sar_building/fcn_r50-d8.py	./work_dirs/fcn_r50-d8/iter_1500.pth	/data2/ghw/sar_building/images/test	/data2/ghw/sar_segm_inference/mixed/fcn

python inference.py ./sar_building/ocrnet_hr48.py	./work_dirs/ocrnet_hr48_512x512_80k_ade20k/iter_1300.pth	/data2/ghw/sar_building/images/test	/data2/ghw/sar_segm_inference/mixed/ocrnet

python inference.py ./sar_building/nonlocal_r50.py	./work_dirs/nonlocal_r50-d8_512x512_80k_ade20k/iter_1600.pth	/data2/ghw/sar_building/images/test	/data2/ghw/sar_segm_inference/mixed/nonlocal

python inference.py ./sar_building/danet_r50.py	./work_dirs/danet_r50-d8_512x512_80k_ade20k/iter_1000.pth	/data2/ghw/sar_building/images/test	/data2/ghw/sar_segm_inference/mixed/danet

python inference.py ./sar_building/deeplabv3p.py	./work_dirs/gf3/deeplabv3plus_r50-d8_512x512_80k_ade20k/iter_300.pth	/home/ghw/mmsegmentation/data/ade20k/sar_building_gf3/images/test	/data2/ghw/sar_segm_inference/gf-gf/deeplabv3p

python inference.py ./sar_building/fcn_r50-d8.py	./work_dirs/gf3/fcn_r50-d8/iter_500.pth	/home/ghw/mmsegmentation/data/ade20k/sar_building_gf3/images/test	/data2/ghw/sar_segm_inference/gf-gf/fcn

python inference.py ./sar_building/ocrnet_hr48.py	./work_dirs/gf3/ocrnet_hr48_512x512_80k_ade20k/iter_800.pth	/home/ghw/mmsegmentation/data/ade20k/sar_building_gf3/images/test	/data2/ghw/sar_segm_inference/gf-gf/ocrnet

python inference.py ./sar_building/nonlocal_r50.py	./work_dirs/gf3/nonlocal_r50-d8_512x512_80k_ade20k/iter_600.pth	/home/ghw/mmsegmentation/data/ade20k/sar_building_gf3/images/test	/data2/ghw/sar_segm_inference/gf-gf/nonlocal

python inference.py ./sar_building/danet_r50.py	./work_dirs/gf3/danet_r50-d8_512x512_80k_ade20k/iter_300.pth	/home/ghw/mmsegmentation/data/ade20k/sar_building_gf3/images/test	/data2/ghw/sar_segm_inference/gf-gf/danet

python inference.py ./sar_building/deeplabv3p.py	./work_dirs/rs2/deeplabv3plus_r50-d8_512x512_80k_ade20k/iter_800.pth	/home/ghw/mmsegmentation/data/ade20k/sar_building_rs2/images/test	/data2/ghw/sar_segm_inference/rs-rs/deeplabv3p

python inference.py ./sar_building/fcn_r50-d8.py	./work_dirs/rs2/fcn_r50-d8/iter_700.pth	/home/ghw/mmsegmentation/data/ade20k/sar_building_rs2/images/test	/data2/ghw/sar_segm_inference/rs-rs/fcn
python inference.py ./sar_building/ocrnet_hr48.py	./work_dirs/rs2/ocrnet_hr48_512x512_80k_ade20k/iter_700.pth	/home/ghw/mmsegmentation/data/ade20k/sar_building_rs2/images/test	/data2/ghw/sar_segm_inference/rs-rs/ocrnet

python inference.py ./sar_building/nonlocal_r50.py	./work_dirs/rs2/nonlocal_r50-d8_512x512_80k_ade20k/iter_800.pth	/home/ghw/mmsegmentation/data/ade20k/sar_building_rs2/images/test	/data2/ghw/sar_segm_inference/rs-rs/nonlocal

python inference.py ./sar_building/danet_r50.py	./work_dirs/rs2/danet_r50-d8_512x512_80k_ade20k/iter_800.pth	/home/ghw/mmsegmentation/data/ade20k/sar_building_rs2/images/test	/data2/ghw/sar_segm_inference/rs-rs/danet