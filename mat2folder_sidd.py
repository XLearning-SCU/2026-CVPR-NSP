import os
import scipy.io as sio
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_sidd_mat_to_png(mat_file_path, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mat_data = sio.loadmat(mat_file_path)

    data_key = None
    for key in mat_data.keys():
        if not key.startswith('__'):
            data_key = key
            break

    if not data_key:
        raise

    images_data = mat_data[data_key]
    print(f".mat file: [{data_key}], shape: {images_data.shape}")

    if len(images_data.shape) == 5:
        n_scenes, n_blocks, h, w, c = images_data.shape
    else:
        raise

    total_images = n_scenes * n_blocks
    with tqdm(total=total_images) as pbar:
        for scene_idx in range(n_scenes):
            for block_idx in range(n_blocks):

                img_array = images_data[scene_idx, block_idx]
                if img_array.dtype != np.uint8:
                    img_array = img_array.astype(np.uint8)
                img = Image.fromarray(img_array)
                
                file_name = f"{scene_idx:04d}_{block_idx:02d}.png"
                save_path = os.path.join(output_dir, file_name)
                img.save(save_path)
                
                pbar.update(1)

    print("Done!\n")

if __name__ == "__main__":
    
    # SIDD Validation Noisy
    noisy_mat_path = "./datasets/SIDD/SIDD_Medium_Srgb/ValidationNoisyBlocksSrgb.mat"
    noisy_out_dir = "./datasets/SIDD/valid_data/sidd_v_noisy"
    if os.path.exists(noisy_mat_path):
        convert_sidd_mat_to_png(noisy_mat_path, noisy_out_dir)
    else:
        print(f"File not found: {noisy_mat_path}")

    print("-" * 40)
    
    # SIDD Validation GT
    gt_mat_path = "./datasets/SIDD/SIDD_Medium_Srgb/ValidationGtBlocksSrgb.mat"
    gt_out_dir = "./datasets/SIDD/valid_data/sidd_v_clean"
    if os.path.exists(gt_mat_path):
        convert_sidd_mat_to_png(gt_mat_path, gt_out_dir)
    else:
        print(f"File not found: {gt_mat_path}")

    print("-" * 40)
    
    # SIDD Benchmark
    benchmark_mat_path = "./datasets/SIDD/SIDD_Medium_Srgb/BenchmarkNoisyBlocksSrgb.mat"
    benchmark_out_dir = "./datasets/SIDD/valid_data/SIDD_Ben"
    if os.path.exists(benchmark_mat_path):
        convert_sidd_mat_to_png(benchmark_mat_path, benchmark_out_dir)
    else:
        print(f"File not found: {benchmark_mat_path}")
