export CUDA_VISIBLE_DEVICES=0

cd /home/csl/code/SAR_build_extract_v2

# RS2 to GF3
# python tools/test.py \
#     configs/FDA/deeplabv3plus_512x512_800_fda_rs2_to_gf3.py \
#     work_dirs/deeplabv3plus_512x512_800_fda_rs2_to_gf3/20210713_155443/latest.pth \
#     --eval mIoU \
#     --show-dir show_dir

# pure RS2
python tools/test.py \
    configs/robustnet/deeplabv3plus_512x512_800_robustnet_rs2.py \
    work_dirs/deeplabv3plus_512x512_800_robustnet_rs2/20210715_154640/latest.pth \
    --eval mIoU \
    --show-dir show_dir