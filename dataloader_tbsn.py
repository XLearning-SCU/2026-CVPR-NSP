import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from natsort import natsorted


def hwc_to_chw(img): # (ps,ps,C)f32
    img = np.transpose(img, [2,0,1]) # (C,ps,ps)f32
    return np.ascontiguousarray(img)

def read_img(filename): # filename:str
	img = cv2.imread(filename) # (H,W,C(bgr))ui8
	img = img[:, :, ::-1] # (H,W,C)f32[0,255]
	return np.array(img).astype('float32') # (H,W,C)f32[0,255]

def get_patch(img, ps): # img:(H,W,C)f32  ps:int
    H, W = img.shape[:2]
    ps_temp = min(H, W, ps)

    xx = 0 if W <= ps_temp else \
        np.random.randint(0, W - ps_temp)
    yy = 0 if H <= ps_temp else \
        np.random.randint(0, H - ps_temp)
    img = img[yy:yy+ps_temp, xx:xx+ps_temp] # (ps,ps,C)f32[0,255]

    if random.randint(0, 1) == 1:
        img = np.rot90(img)
    if random.randint(0, 1) == 1:
        img = np.flip(img, axis=1)
    if random.randint(0, 1) == 1:
        img = np.flip(img, axis=0)

    return img


class DatasetForTrainingMEM(Dataset):
    def __init__(self, noise_imgs, patch_sz):
        super().__init__()
        self.patch_sz = patch_sz
        self.noise_imgs = noise_imgs#[:8]

        self.img_num = len(self.noise_imgs)
        self.img_num_fake = int(self.img_num * 20)
        self.NOI = []
        cnt = 0
        for noi_path in self.noise_imgs:
            noi = cv2.imread(noi_path)[:,:,::-1] # (H,W,C)ui8
            noi = noi.copy().astype(np.float32)#/255.0 # (H,W,C)f32[0,1]
            self.NOI.append(noi) # (H,W,C)f32[0,1]
            cnt += 1
            print("Loading %4d/%4d training imgs.\r"%(cnt,self.img_num), end="")
        print("\n")
    
    def __len__(self):
        return self.img_num_fake

    def __getitem__(self, index):
        index %= self.img_num
        img = self.NOI[index] # (H,W,C)f32[0,1]
        img = get_patch(img, self.patch_sz) # (ps,ps,C)f32[0,1]  50%转90度 50%水平翻转 50%竖直翻转
        return hwc_to_chw(img) # (C,ps,ps)f32[0,1]


class DatasetForValidation(Dataset):
    def __init__(self, noise_imgs, clean_imgs, saved_imgs):
        super().__init__()
        self.noise_imgs = noise_imgs#[:3]
        self.clean_imgs = clean_imgs#[:3]
        self.saved_imgs = saved_imgs#[:3]

    def __len__(self):
        return len(self.noise_imgs)

    def __getitem__(self, index):
        noise_img = self.noise_imgs[index]
        noise_img = hwc_to_chw(read_img(noise_img))
        
        clean_img = hwc_to_chw(read_img(self.clean_imgs[index])) \
            if len(self.clean_imgs) == len(self.noise_imgs) else 0
        
        saved_img = self.saved_imgs[index] \
            if len(self.saved_imgs) == len(self.noise_imgs) else ''
        
        return noise_img, clean_img, saved_img

from glob import glob
def loading_datasets(train_dir, val_rn_dir, val_gt_dir, save_dir, bs, ps, dist, world_size, num_workers):

    ''' Train '''
    # train_dir: "....../*.PNG"
    train_imgs = natsorted( glob( train_dir ) )#[:bs]
    train_dataset = DatasetForTrainingMEM(train_imgs, ps)
    dist_sampler = DistributedSampler(train_dataset) if dist else None
    train_loader = DataLoader(train_dataset, batch_size=bs//world_size if dist else bs,
                              shuffle=False if dist else True, sampler=dist_sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

    ''' Validation '''
    test_noise_imgs = natsorted(glob(val_rn_dir))#[43:45]
    test_clean_imgs = natsorted(glob(val_gt_dir))#[43:45]
    test_saved_imgs = [ os.path.join(save_dir, img.split("/")[-1]) for img in test_noise_imgs ]#[43:45]
    
    val_dataset = DatasetForValidation(test_noise_imgs, test_clean_imgs, test_saved_imgs)
    validate_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    return train_loader, validate_loader, dist_sampler