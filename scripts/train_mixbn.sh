cd /home/csl/code/SAR_build_extract_v2

# codna init
# conda activate sar_build_extract_v2

# floating point arithmetic operations are not support by vanilla bash
# for rt in {0..10..2}
for rt in 0 0.25 0.5 0.75 1
do 
# echo $((rt/10.))
python tools/train.py configs/mix_bn/deeplabv3plus_512x512_800_mixbn_rs2_to_gf3.py --options {model.backbone.norm_cfg.ratio=$rt,model.decode_head.norm_cfg.ratio=$rt,model.auxiliary_head.norm_cfg.ratio=$rt}
done

