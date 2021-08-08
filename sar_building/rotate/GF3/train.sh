cd /home/ghw/mmsegmentation/
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/rotate/GF3/danet_r50.py 2
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/rotate/GF3/deeplabv3p.py 2
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/rotate/GF3/fcn_r50-d8.py 2
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/rotate/GF3/ocrnet_hr48.py 2
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh sar_building/rotate/GF3/pspnet_r50-d8.py 2