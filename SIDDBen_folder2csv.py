
import os
import numpy as np
import pandas as pd
import base64
import sys
import torch

from natsort import natsorted
import cv2


def array_to_base64string(x): # x:(H,W,C)ui8
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string


def base64string_to_array(base64string, array_dtype, array_shape):
    decoded_bytes = base64.b64decode(base64string)
    decoded_array = np.frombuffer(decoded_bytes, dtype=array_dtype)
    decoded_array = decoded_array.reshape(array_shape)
    return decoded_array


def main():

    de_dir = sys.argv[1] # "../TBSN-aaai-25/results/tbsn_r3_siddben"
    de_imgs = natsorted(os.listdir(de_dir))

    output_blocks_base64string = []
    cnt = 0
    with torch.no_grad():
        for de_img in de_imgs:
            out_block = cv2.imread(os.path.join(de_dir, de_img))[:,:,::-1].astype(np.uint8) # (H,W,C)f32[0,255]
            out_block_base64string = array_to_base64string(out_block)
            output_blocks_base64string.append(out_block_base64string)

            cnt += 1
            print("{:4d}/{:4d} have been denoised.\r".format(cnt, 1280), end="")

    # Save outputs to .csv file.
    output_file = de_dir.split("/")
    output_file.append("SubmitSrgb.csv")
    # output_file = pth_dir.split("/")
    # output_file[-1] = "SubmitSrgb.csv" # 替换掉最后的.pth文件地址
    output_file = "/".join(output_file) # 还原成路径的str格式
    print(f'Saving outputs to {output_file}')
    output_df = pd.DataFrame()
    n_blocks = len(output_blocks_base64string)
    print(f'Number of blocks = {n_blocks}')
    output_df['ID'] = np.arange(n_blocks)
    output_df['BLOCK'] = output_blocks_base64string

    output_df.to_csv(output_file, index=False)

    # TODO: Submit the output file SubmitSrgb.csv at 
    # kaggle.com/competitions/sidd-benchmark-srgb-psnr
    print('TODO: Submit the output file SubmitSrgb.csv at')
    print('kaggle.com/competitions/sidd-benchmark-srgb-psnr')

    print('Done.')


if __name__ == "__main__":
    main()
    