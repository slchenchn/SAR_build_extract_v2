cd /home/csl/code/SAR_build_extract_v2

# RS2 to GF3
# CUDA_VISIBLE_DIVICES=0,1 ./tools/dist_train.sh configs/robustnet/deeplabv3plus_512x512_800_robustnet_rs2.py 2

# pure RS2
CUDA_VISIBLE_DIVICES=2,3 PORT=29502 ./tools/dist_train.sh configs/robustnet/deeplabv3plus_512x512_800_robustnet_rs2.py 2

