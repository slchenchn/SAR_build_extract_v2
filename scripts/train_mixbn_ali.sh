cd /home/csl/code/SAR_build_extract_v2

# codna init
# conda activate sar_build_extract_v2

# floating point arithmetic operations are not support by vanilla bash

# whole model
# for rt in 0 0.25 0.5 0.75 
# do 
# # echo $((rt/10.))
# python tools/train.py configs/mix_bn/deeplabv3plus_512x512_4k_mixbn_rs2_to_gf3_0_detach.py --options {model.backbone.norm_cfg.ratio=$rt,model.decode_head.norm_cfg.ratio=$rt,model.auxiliary_head.norm_cfg.ratio=$rt}
# done

# only backbone
for rt in 0.25 0.5 0.75 
do 
# echo $((rt/10.))
python tools/train.py configs/mix_bn/deeplabv3plus_512x512_4k_mixbn_rs2_to_gf3_2_detach_backbone.py --options model.backbone.norm_cfg.ratio=$rt
done

