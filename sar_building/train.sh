cd /home/ghw/mmsegmentation/
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/gf3/deeplabv3p.py 2
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/gf3/fcn_r50-d8.py 2
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/gf3/ocrnet_hr48.py 2
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/gf3/nonlocal_r50.py 2
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/gf3/danet_r50.py 2
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/rs2/deeplabv3p.py 2
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/rs2/fcn_r50-d8.py 2
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/rs2/ocrnet_hr48.py 2
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/rs2/nonlocal_r50.py 2
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/rs2/danet_r50.py 2