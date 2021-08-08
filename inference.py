from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os
import numpy as np
import tqdm
import argparse

def show_result(result,
                palette=None):
    seg = result[0]
    palette = np.array(palette)
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    img = color_seg
    img = img.astype(np.uint8)
    return img



parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("pth")
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()
input_path = args.input
output_path = args.output
os.makedirs(output_path,exist_ok=True)
img_list = os.listdir(input_path)

# build the model from a config file and a checkpoint file
model = init_segmentor(args.config, args.pth, device="cuda:0")
for i in tqdm.tqdm(img_list):
    img_path = os.path.join(input_path, i)
    result = inference_segmentor(model, img_path)
    img = show_result(result, palette=model.PALETTE)
    # img = img[:, :, 0] / 255
    mmcv.imwrite(img, os.path.join(output_path, i.split(".")[0] + ".jpg"))
