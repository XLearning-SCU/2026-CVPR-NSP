import os
import h5py
import numpy as np
import cv2

data_dir = './datasets/dnd_2017' 
output_dir = './datasets/DND_folder'
os.makedirs(output_dir, exist_ok=True)

info_path = os.path.join(data_dir, 'info.mat')

print("正在读取 info.mat ...")
with h5py.File(info_path, 'r') as info_file:
    info = info_file['info']
    bb = info['boundingboxes']

    for i in range(50):
        img_idx = i + 1 
        
        mat_filename = f'{img_idx:04d}.mat'
        mat_filepath = os.path.join(data_dir, 'images_srgb', mat_filename)
        
        with h5py.File(mat_filepath, 'r') as img_data:
            Inoisy = np.array(img_data['InoisySRGB']).T
        
        ref = bb[0][i]
        boxes = np.array(info_file[ref]).T
        
        for k in range(20):
            patch_idx = k + 1
            
            min_h = int(boxes[k, 0]) - 1
            max_h = int(boxes[k, 2])
            min_w = int(boxes[k, 1]) - 1
            max_w = int(boxes[k, 3])
            
            patch = Inoisy[min_h:max_h, min_w:max_w, :]
            
            patch_uint8 = np.clip(np.floor(patch * 255.0 + 0.5), 0, 255).astype(np.uint8)
            
            patch_bgr = cv2.cvtColor(patch_uint8, cv2.COLOR_RGB2BGR)
            
            out_filename = f'{img_idx:04d}_{patch_idx:02d}.png'
            out_filepath = os.path.join(output_dir, out_filename)
            
            cv2.imwrite(out_filepath, patch_bgr)
            
        print(f"Extraction complete for image: {img_idx:02d}/50")

print("Done!")