import math
import time
import torch
# import wandb
import numpy as np
import argparse, os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
import random

from TBSN import TBSN_NSP
from dataloader_tbsn import loading_datasets
from utils import AverageMeter, batch_SSIM, batch_PSNR
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default="./datasets/SIDD/SIDD_Medium_Srgb/Data/*/*_NOISY_*.PNG")
parser.add_argument('--val_rn_dir', type=str, default="./datasets/SIDD/valid_data/sidd_v_noisy/*.png")
parser.add_argument('--val_gt_dir', type=str, default="./datasets/SIDD/valid_data/sidd_v_clean/*.png")

parser.add_argument('--ckpts', type=str, default='./ckpts')
parser.add_argument('--imgs', type=str, default='./imgs')
parser.add_argument('--pretrained_model', type=str, default='')

parser.add_argument('--bs', type=int, default=4, help='batch size')
parser.add_argument('--ps', type=int, default=640, help='patch size')
parser.add_argument('--tds', type=int, default=5, help='training stride')
parser.add_argument('--ids', type=int, default=2, help='inference stride')
parser.add_argument('--ng', type=int, default=1, help='')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=900, help='training epochs')
parser.add_argument('--sr_rec', type=bool, default=False, help='sr reconstruction')
parser.add_argument('--chosen', type=int, default=1, help="randomly choose 'chosen' imgs form p*p sub-imgs")
parser.add_argument('--seed', type=int, default=10)

parser.add_argument('--world_size', type=int, default=1, help='number of processes')
parser.add_argument('--local_rank', type=int, default=0, help='local rank of process')
parser.add_argument('--dist', type=bool, default=False, help='activate distributed parallel')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--gpuid', type=str, default="0")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid


if torch.cuda.device_count() > 1 and args.dist==True:
    # args.dist = True
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(f'cuda:{args.local_rank}')
    torch.distributed.init_process_group('nccl')
    args.world_size = torch.distributed.get_world_size()
    if args.local_rank == 0:
        print('enable distributed parallel training')


if args.local_rank == 0:
    # wandb.init(project="SDN")
    psnr_max = 0.0
    ssim_par = 0.0


def pixel_unshuffle(x, f:int=2): # x:[B,C,H,W]
    sz = x.size()[-2:]
    p_h, p_w = [math.ceil(s / f) * f - s for s in sz]
    x = x if p_h==0 and p_w==0 else F.pad(x, (0, p_w, 0, p_h), 'reflect')

    B, C, H, W = x.size()
    h, w = H // f, W // f
    x = x.contiguous().view(B, C, h, f, w, f) # [B,C,H/f,f, W/f,f]
    y = x.permute(0, 3, 5, 1, 2, 4).contiguous() # [B,f,f, C, H/f, W/f]
    y = y.view(B * f * f, C, h, w) # [B*f*f, C, H/f, W/f]

    return y, sz


def pixel_shuffle(x, sz, sr_rec, f):
    B, C, H, W = x.size()
    f2 = f*f
    B = B // f2
    if sr_rec: # x:[B*ids*ids, C, H/ids, W/ids]  sz=torch.Shape([H/ids, W/ids])
        x = x.view(B, f, f, C, H, W) # [B,ids,ids, C, H/ids, W/ids]
        y = x.permute(0, 3, 4, 1, 5, 2).contiguous() # [B, C, H/ids,ids, W/ids,ids]
        y = y.view(B, C, H*f, W*f) # [B,C,H,W]
        return y[..., :sz[0]*f, :sz[1]*f] # [B,C,H,W]
    else: # x:[B*ids*ids, C, H, W]  sz=torch.Shape([H, W])
        x = x.view(B, f2, C, H, W) # [B, ids*ids, C,H,W]
        return x.mean(dim=1)[..., :sz[0], :sz[1]] # [B,C,H,W]


def test(val_loader, model):
    model = model.module if args.dist else model
    
    model.eval()
    nums = len(val_loader)
    psnrs, ssims, imgs = [], [], []
    
    for idx, (input, label, save_way) in enumerate(val_loader):
        with torch.no_grad():
            input, sz = pixel_unshuffle(input.cuda(), f=args.ids) # [1*ids*ids, C, H/2/ids,W/2/ids]f32[0,1]  sz=torch.Size([H/2, W/2])
            im_denoised = pixel_shuffle(model(input), sz, args.sr_rec, f=args.ids) # 

        label /= 255.0
        im_denoised /= 255.0
        im_denoised.clamp_(0.0, 1.0)

        if idx==310:
            imgs.append((idx, save_way[0], im_denoised))

        if im_denoised.size() == label.size():
            psnrs.append( batch_PSNR(im_denoised, label) )
            ssims.append( batch_SSIM(im_denoised, label) )

    return psnrs, ssims, imgs

''' x [ids,ids] '''
# import itertools
# rows_combs = list(itertools.combinations(range(args.tds), args.ids))   # C(p,s) 个
# cols_combs = list(itertools.combinations(range(args.tds), args.ids))   # C(p,s) 个
# candidate_len = 0 # len=C(p,s)*C(p,s)
# candidate_idxsets = []  # 每个候选对对应的平面内下标集合（扁平化的 0..p2-1）
# for r in rows_combs:
#     for c in cols_combs:
#         # 计算交叉位置的扁平下标
#         # 每 pos = row * p + col
#         idxs = []
#         for rr in r:
#             for cc in c:
#                 idxs.append(rr * args.tds + cc)
#         idxs = np.array(idxs, dtype=np.int64)
#         candidate_len += 1
#         candidate_idxsets.append(idxs)
# candidate_idxsets = np.array(candidate_idxsets, dtype=np.int64) # (C(p,s)*C(p,s), s*s)i64 所有可能的[s,s]下表

''' continuous [ids,ids] '''
topleft_pos = [(r,c) for r in range(args.tds-args.ids+1) for c in range(args.tds-args.ids+1)]
candidate_len = 0
candidate_idxsets = []
for pos in topleft_pos:
    r_start, c_start = pos
    idxs = []
    for rr in range(r_start, r_start + args.ids):
        for cc in range(c_start, c_start + args.ids):
            idxs.append(rr * args.tds + cc)
    idxs = np.array(idxs, dtype=np.int64)
    candidate_len += 1
    candidate_idxsets.append(idxs)


def generate_mask_pair(img):  # img:[n,c,h,w]f32[0,1]
    n_batch, _, h, w = img.shape
    p, s = args.tds, args.ids
    p2 = int(p * p)

    # 最大不重叠组数（理论上）：floor(p/s) * floor(p/s)
    req_n = args.ng

    # masks 初始化，与原来一致
    masks = [torch.zeros(
        (n_batch * (h // p) * (w // p) * (p2),),
        dtype=torch.bool, device=img.device) for _ in range(p2)]  # [p^2][nhw]

    num_blocks = n_batch * (h // p) * (w // p)  # nhw/p^2 的数量乘以 p^2 后的计数基数
    # rd_idx 将是 (num_blocks, p2)
    rd_idx = np.zeros([num_blocks, p2], dtype=np.int64) # [nhw/p^2, p^2]

    # 2) 对每个 block 单独随机选择 req_n 个互不相交的 candidate
    # 为了随机性，先生成一个随机排列的 candidate 索引列表，然后贪心选择（跳过与已选集合有交集的候选）
    cand_indices_master = np.arange(candidate_len, dtype=np.int64) # (C(p,s)*C(p,s), )i64

    for blk in range(num_blocks): # nhw/p^2
        # 随机顺序（每个 block 独立）
        np.random.shuffle(cand_indices_master)
        selected_idx_list = []
        used_mask = np.zeros(p2, dtype=bool) # (p*p,)标记已被占用的位置
        for cand_idx in cand_indices_master:
            cand_set = candidate_idxsets[cand_idx] # (s*s,): label的s*s个下标
            # 如果与已有选中的集合有交集，则跳过
            if used_mask[cand_set].any():
                continue
            # 否则选中并标记
            selected_idx_list.append(cand_set)
            used_mask[cand_set] = True
            if len(selected_idx_list) >= req_n:
                break

        # 如果没能选够 req_n（理论上不应发生，因为我们在开始就限制 req_n <= max_groups），
        # 则抛出异常（以便立即发现配置/逻辑问题）
        if len(selected_idx_list) < req_n:
            raise RuntimeError(f"Could not find {req_n} non-overlapping groups for block {blk}. p={p}, s={s}")

        # 把选中的所有 idx 拼成一维数组（保持每组 s*s 的内部顺序不变）
        front_idx = np.concatenate(selected_idx_list).astype(np.int64)  # 长度 = req_n * s*s

        # 构造 array: 先把 front_idx 放前面，剩下的按照随机顺序放后面（与原逻辑一致）
        array = np.arange(p2, dtype=np.int64)
        mask = np.ones(p2, dtype=bool)
        mask[front_idx] = False
        tail = array[mask]
        np.random.shuffle(tail)
        arr = np.concatenate((front_idx, tail))  # 长度 p2
        rd_idx[blk, :] = arr

    # 转为 torch 并添加偏移（与原逻辑一致）
    rd_idx_t = torch.from_numpy(rd_idx).type(torch.int64).to(img.device)
    rd_idx_t += torch.arange(0, num_blocks * p2, step=p2,
                             dtype=torch.int64, device=img.device).reshape(-1, 1)

    # 生成 masks
    for i in range(len(masks)):
        masks[i][rd_idx_t[:, i]] = 1

    return masks # [p^2, nhw]


def generate_mask_pair_n_sort(img): # img:[n,c,h,w]f32[0,1]
    n, _, h, w = img.shape
    p, s, ng = args.tds, args.ids, args.ng
    p2 = int(p*p)
    s2 = int(s*s)

    masks = [torch.zeros(
        (n * (h // p) * (w // p) * p2,),
        dtype=torch.bool, device=img.device) for _ in range(p2)] # [p^2][nhw]
    rd_idx = np.zeros([n * (h // p) * (w // p), p2]) # (nhw/p^2, p^2)f64

    for i in range(rd_idx.shape[0]): # nhw/p^2
        array = np.arange(rd_idx.shape[1]) # (0, ..., p^2-1)i64
        np.random.shuffle(array)
        # for j in range(ng):
        #     start = j*s2
        #     end = start + s2
        #     array[start:end] = np.sort(array[start:end])
        rd_idx[i,:] = array # [p^2]
    rd_idx = torch.from_numpy(rd_idx).type(torch.int64).to(img.device) # [nhw/p^2, p^2]i64  each row: rdperm[0...p^2-1]
    rd_idx += torch.arange(0, n*(h//p)*(w//p)*(p2), step=(p2),
                           dtype=torch.int64, device=img.device).reshape(-1, 1) # [nhw/p^2, p^2]i64  add the offsets

    for i in range(len(masks)):
        masks[i][rd_idx[:,i]] = 1 # each row[nhw] has nhw/p^2 1s, other places all 0.
    return masks # [p^2, nhw]


def generate_imgs_pair(img, masks): # img:[b,c,h,w]  masks:[p^2][nhw]  only nhw/p^2 ones in each row[nhw]
    b, c, h, w = img.shape
    p, s = args.tds, args.ids
    n = args.ng
    nss = n*s*s
    p2 = len(masks)

    subimages = torch.zeros(
        b, p2, c, h // p, w // p, 
        dtype=img.dtype, device=img.device) # [b, p^2, c, h/p, w/p]f32
    
    for i in range(p2): # [p^2]
        for j in range(c):
            img_per_ch = F.unfold(img[:, j:j+1, :, :], p, stride=p) # [b,1,h,w]->[b, p^2, hw/p^2]
            img_per_ch = img_per_ch.view(b, p * p, h // p, w // p) # ->[b, p^2, h/p, w/p]
            img_per_ch = img_per_ch.permute(0, 2, 3, 1).reshape(-1) # ->[b, h/p, w/p, p^2] ->[bhw]  行优先遍历img的每个[p,p],再右面的[p,p]...
            img_per_ch = img_per_ch[masks[i]].reshape(b, h // p, w // p, 1) # [bhw]->[bhw/p^2]->[b, h/p, w/p, 1]  img的每个[p,p]中随机选了一个像素
            subimages[:, i, j:j+1, :, :] = img_per_ch.permute(0, 3, 1, 2) # [b, 1, h/p, w/p]

    # subimages:[b, p^2, c, h/p, w/p]
    noise_sup = subimages[:,:nss].view(b,n, s,s, c, h // p, w // p) # [b, n*s^2, c, h/p, w/p]->[b,n,s,s,c,h/p,w/p]
    noise_sup = noise_sup.permute(0,1, 4, 5, 2, 6, 3).contiguous() # [b,n,c, h/p,s, w/p,s]
    noise_sup = noise_sup.view(b, n, c, h // p * s, w // p * s) # [b,n,c, h/p*s, w/p*s]
    noise_sup = noise_sup.repeat(1, (p2-nss), 1, 1, 1) # [b, (p^2-n*s^2)*n, c, sh/p, sw/p]
    noise_sup = noise_sup.view(-1, c, h // p * s, w // p * s) # [b*n*(p^2 - n*s^2), c, h/p*s, w/p*s]

    noise_sub = subimages[:,nss:].unsqueeze(2).repeat(1,1,n,1,1,1).reshape(-1, c, h // p, w // p) # [b*n*(p^2 - n*s^2), c, h/p, w/p]
    # noise_sub = subimages.unsqueeze(2).repeat(1,1,n,1,1,1).reshape(-1,c, h//p, w//p) # [b,p2,1->n,c,h/p,w/p]->[b*p2*n,c,h/p,w/p]
    return noise_sub.contiguous(), noise_sup.contiguous() # [b*n*(p^2 - n*s^2), c, h/p, w/p]  [b*n*(p^2 - n*s^2), c, h/p*s, w/p*s]


def train(train_loader, val_loader, model, criterion, optimizer, epoch):
    if args.local_rank == 0:
        global psnr_max, ssim_par
        losses = AverageMeter()
        start_time = time.time()

    LEN_1eph = len(train_loader)
    for i, img in enumerate(tqdm(train_loader)): # img:[B,C,H,W]f32[0,1]
        model.train()
        img = img.cuda() # [B,C,H,W]f32[0,1]
        optimizer.zero_grad()

        # masks = generate_mask_pair_n_sort(img) # [p^2][nhw] each row[nhw] only has nhw/p^2 1s.
        masks = generate_mask_pair(img)
        masks = masks[:(args.ids*args.ids+args.chosen)]
        lr,hr = generate_imgs_pair(img, masks) # [b(p^2-s^2), c, h/p, w/p]f32[0,1]  [b(p^2-s^2), c, h/p*s, w/p*s]f32[0,1]
        de = model(lr)
        loss = criterion(de, hr)

        loss.backward()
        optimizer.step()

        if args.dist: 
            torch.distributed.all_reduce(
                loss.clone().detach(), 
                op=torch.distributed.ReduceOp.SUM)

        if args.local_rank == 0:
            losses.update(loss.item() /args.world_size)

    if args.local_rank==0 and epoch%5==0:
        psnrs, ssims, imgs = test(val_loader, model)
        n_img = len(psnrs)
        avg_psnr = sum(psnrs)/n_img
        avg_ssim = sum(ssims)/n_img
        if avg_psnr >= psnr_max:
            psnr_max = avg_psnr
            ssim_par = avg_ssim

        for idx, way, img in imgs:
            if args.imgs in way:
                wa = os.path.dirname(way)
                if not os.path.isdir(wa):
                    os.makedirs(wa)
                save_image(img, os.path.join(args.imgs, "%s_eph%d_%d_%d.png"%(way.split("/")[-1][:-4], epoch, round(psnrs[idx]*100), round(ssims[idx]*10000))))
        v = '_ckpt_it{:d}_{:d}_{:d}.pth'.format(epoch, round(avg_psnr*100), round(avg_ssim*10000))
        # state_dict = model.module.state_dict() if args.dist else model.state_dict()
        # torch.save(state_dict, os.path.join(args.ckpts, v))
        state = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if args.dist else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'cuda_rng_state_all': torch.cuda.get_rng_state_all(), # for multi-gpu
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate()
        }
        torch.save(state, os.path.join(args.ckpts, v))

        run_time = int(time.time() - start_time)
        # wandb.log({"Loss": losses.avg, "SSIM": ssim, "PSNR": psnr, "PSNR_max": psnr_max}, step=epoch)
        content = 'Eph:{:d}, {:4.2f}/{:5.4f}, max: {:4.2f}/{:5.4f}, time={:.1f}, loss={:.4e}'.format(epoch, avg_psnr, avg_ssim, psnr_max, ssim_par, run_time, loss)
        print(content)
        f = open(os.path.join(args.ckpts, "../raw.txt"), "a+")
        f.write(content + "\n")
        f.close()
    print("{:4d}/{:4d}\r".format(i+1, LEN_1eph), end="")


def set_seeds(seed:int=0):
    random.seed(seed)
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # torch cpu
    torch.cuda.manual_seed(seed) # torch gpu
    torch.cuda.manual_seed_all(seed) # torch multi-gpu
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


if __name__ == '__main__':
    set_seeds(args.seed)
    if args.local_rank == 0 and not os.path.isdir(args.ckpts):
        os.makedirs(args.ckpts)
    if args.local_rank == 0 and not os.path.isdir(args.imgs):
        os.makedirs(args.imgs)

    f = open(os.path.join(args.ckpts, "../raw.txt"), "a+")
    for arg in str(args).split(", "):
        f.write(arg+"\n")
    f.write("\n\n") 
    f.close()

    criterion = nn.L1Loss().cuda()

    model = TBSN_NSP(ids=args.ids).cuda() if not args.dist else \
        DDP(TBSN_NSP(ids=args.ids).cuda(), [torch.cuda.current_device()])
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    if os.path.exists(args.pretrained_model):
        checkpoint = torch.load(args.pretrained_model, map_location='cpu')

        if not args.dist:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        random.setstate(checkpoint['random_rng_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
        np.random.set_state(checkpoint['numpy_rng_state'])

        epoch_start = checkpoint['epoch'] + 1
        if args.local_rank == 0:
            print(f"loaded model and states from '{args.pretrained_model}'")
    else:
        epoch_start = 0

    if args.local_rank == 0: 
        print(f"loading datasets '{args.train_dir}'")

    train_loader, val_loader, dist_sampler = loading_datasets(
        args.train_dir, args.val_rn_dir, args.val_gt_dir, args.imgs, args.bs, args.ps, 
        args.dist, args.world_size, args.num_workers)

    if args.local_rank == 0:
        print(f'loaded training images: {len(train_loader.dataset)}')
        print(f'loaded validation images: {len(val_loader.dataset)}')
        print('start training models ...')

    for epoch in range(epoch_start, args.epochs + 1):
        dist_sampler.set_epoch(epoch) if args.dist else None
        train(train_loader, val_loader, model, criterion, optimizer, epoch)

    # wandb.finish()