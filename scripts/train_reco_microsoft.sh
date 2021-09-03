cd /home/chenshuailin/code/SAR_build_extract_v2


# python tools/train.py configs/reco/deeplabv3plus_reco_8k_s_w_sar_build.py --options model.apply_pseudo_loss=false

# python tools/train.py configs/reco/deeplabv3plus_reco_8k_s_w_sar_build.py --options model.apply_reco=false


# python tools/train.py configs/reco/deeplabv3plus_reco_sar_build.py --options model.apply_pseudo_loss=false model.apply_reco=false


python tools/train.py configs/reco/deeplabv3plus_reco_8k_s_s_sar_build.py --options model.apply_pseudo_loss=false

python tools/train.py configs/reco/deeplabv3plus_reco_8k_s_s_sar_build.py --options model.apply_reco=false

