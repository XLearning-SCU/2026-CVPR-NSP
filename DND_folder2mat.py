import numpy as np
import scipy.io as sio
import os
import cv2
from natsort import natsorted
import sys

def main():
    data_folder = sys.argv[1] # "../TBSN-aaai-25/results/tbsn_r3_dnd"
    imgs_paths = natsorted(os.listdir(data_folder))
    out_folder = os.path.join(data_folder, "dnd_submit_files")
    os.makedirs(out_folder, exist_ok=True)

    # process data
    for i in range(50):
        for k in range(20):
            global_idx = i*20 + k
            de = cv2.imread(os.path.join(data_folder, imgs_paths[global_idx])) # ( H,W,C(bgr) )ui8
            de = de[:,:,::-1].astype(np.float32)/255.0 # (H,W,C)f32[0,1]

            Idenoised_crop = de # (P,P,C)<f4[0,1]
            # save denoised data
            Idenoised_crop = np.float32(Idenoised_crop) # (P,P,C)f32[0,1]
            save_file = os.path.join(out_folder, '%04d_%02d.mat'%(i+1,k+1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
            print('%d/%d\r' % (k+1, 20), end="")
        print('[%d/%d] done\n' % (i+1, 50))

    ''' bundle '''
    submission_folder = out_folder
    out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:pass
    israw = False
    eval_version="1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=object)
        for bb in range(20):
            filename = '%04d_%02d.mat'%(i+1,bb+1)
            s = sio.loadmat(os.path.join(submission_folder, filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat'%(i+1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )
    for file_name in os.listdir(submission_folder):
        if file_name.endswith('.mat'):
            os.remove(os.path.join(submission_folder, file_name))


if __name__=="__main__":
    main()
