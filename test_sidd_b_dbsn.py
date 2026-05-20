import os
import math
import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from DBSNl import DBSNl
from dataloader import SIDD_Ben_folder
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str, default='datasets/SIDD/valid_data/SIDD_Ben') # SIDD Ben
parser.add_argument('--ids', type=int, default=2, help='test stride')
parser.add_argument('--model', type=str, default='./ckpts/nsp_dbsn_noseed_it620_3702_8865.pth',)
parser.add_argument('--sr_rec', type=bool, default=False, help='sr reconstruction')
args = parser.parse_args()

from datetime import datetime
now = datetime.now()
args.result_dir = "results/"+now.strftime("%Y-%m-%d_%H-%M-%S")+"-siddben"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pixel_unshuffle(x): # x:[B,C,H,W]
    sz = x.size()[-2:]
    p_h, p_w = [math.ceil(s / args.ids) \
                * args.ids - s for s in sz]
    x = x if p_h==0 and p_w==0 else \
        F.pad(x, (0, p_w, 0, p_h), 'reflect')

    B, C, H, W = x.size()
    h, w = H // args.ids, W // args.ids
    x = x.contiguous().view(B, C, h, args.ids, w, args.ids) # [B, C, H/2, 2, W/2, 2]
    y = x.permute(0, 3, 5, 1, 2, 4).contiguous() # [B,2,2, C, H/2, W/2]
    y = y.view(B * args.ids * args.ids, C, h, w) # [B*2*2, C, H/2, W/2]

    return y, sz # [B*4, C, H/2, W/2]  (H,W)

def pixel_shuffle(x, sz, sr_rec): # x:[B,C,H,W]  (H,W)
    B, C, H, W = x.size()
    B = B // (args.ids * args.ids)
    if sr_rec:
        x = x.view(B, args.ids, args.ids, C, H, W)
        y = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        y = y.view(B, C, H * args.ids, W * args.ids)
        return y[..., :sz[0] * args.ids, :sz[1] * args.ids]
    else:
        x = x.view(B, args.ids * args.ids, C, H, W) # [B, 4, C, H, W]
        return x.mean(dim=1)[..., :sz[0], :sz[1]] # [B,C,H,W]

def test(loader, model):
    model.eval()
    cnt = 0
    os.makedirs(args.result_dir, exist_ok=True)

    for input, save_way in tqdm(loader):
        B,C,H,W = input.shape
        with torch.no_grad():
            input = input.to(device)
            ''' use PD '''
            im, sz = pixel_unshuffle(input) # [B*4, C, H/2, W/2]  (H,W)
            im_denoised = pixel_shuffle(model(im), sz, args.sr_rec) # [B*4, C, H, W]-avg->[B,C,H,W]
            if args.sr_rec:
                sr_pd, sz2 = pixel_unshuffle(im_denoised) # [B*4, C, H/2, W/2]
                im_denoised = pixel_shuffle(model(sr_pd), sz2, sr_rec=False) # ->[B*4,C,H,W]->[B,C,H,W]
            ''' direct (better) '''
            # im_denoised = model(input)

        im_denoised = torch.floor(im_denoised*255+0.5).clamp(0.0, 255.0)/255.0

        wa = os.path.dirname(save_way[0])
        b1 = cnt//32
        b2 = cnt%32
        save_image(im_denoised, os.path.join(wa,"{:04d}_{:02d}.png".format(b1,b2)))
        cnt += 1
    return None


if __name__ == '__main__':
    model = DBSNl(ids=args.ids).to(device)
    checkpoint = torch.load(args.model)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    nparas = 0
    for m in model.parameters():
        if m.requires_grad:
            nparas += m.numel()
    print(f"loaded model '{args.model}'  nparas={nparas}")

    imgs = sorted(os.listdir(args.in_dir))
    noise_imgs = [os.path.join(args.in_dir, img) for img in imgs]
    saved_imgs = [] if args.result_dir == '' else \
        [os.path.join(args.result_dir, img) for img in imgs]
    testset = SIDD_Ben_folder(noise_imgs, saved_imgs)

    print('start testing model ...')
    test(DataLoader(testset, 1), model)
