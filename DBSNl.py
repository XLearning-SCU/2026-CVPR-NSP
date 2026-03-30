import torch
import torch.nn as nn

class DBSNl(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included. 
    see our supple for more details. 
    '''
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9, ids:int=2):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divided with 2"

        ly = []
        ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1, bias=True) ]
        ly += [ nn.GELU() ]
        self.head = nn.Sequential(*ly)

        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)

        ly = []
        ly += [ nn.Conv2d(base_ch*2,  base_ch*2,    kernel_size=1, bias=True) ]
        ly += [ nn.GELU() ]
        ly += [ nn.Conv2d(base_ch*2,  base_ch*2,    kernel_size=1, bias=True) ]
        ly += [ nn.GELU() ]
        #
        ly += [ nn.PixelShuffle(ids) ] # [B,2D,H,W] -> [B, 2D/ids^2, H*ids,W*ids]
        #
        ly += [ nn.Conv2d(base_ch*2//(ids*ids), base_ch*2//(ids*ids), kernel_size=1, bias=True) ]
        ly += [ nn.GELU() ]
        ly += [ nn.Conv2d(base_ch*2//(ids*ids), base_ch*2//(ids*ids), kernel_size=1, bias=True) ]
        ly += [ nn.GELU() ]
        ly += [ nn.Conv2d(base_ch*2//(ids*ids), out_ch,     kernel_size=1, bias=True) ]
        # ly += [ nn.PixelShuffle(2) ] # [B,2D,H,W] -> [B, D/2, 2H,2W]
        # ly += [ nn.Conv2d(base_ch//2,    base_ch//2, kernel_size=1, bias=True) ]
        # ly += [ nn.GELU() ]
        # ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1, bias=True) ]
        # ly += [ nn.GELU() ]
        # ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1, bias=True) ]
        self.tail = nn.Sequential(*ly)

    def forward(self, x): # x:[B,C,H,W]
        x = self.head(x) # [B,D,H,W]

        br1 = self.branch1(x) # still [B,D,H,W]
        br2 = self.branch2(x) # still [B,D,H,W]

        x = torch.cat([br1, br2], dim=1) # [B,2D,H,W]

        return self.tail(x) # [B,C,2H,2W]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [ CentralMaskedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1, bias=True) ]
        # ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=2*stride-1, padding=stride-1)]
        ly += [ nn.GELU() ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True) ]
        ly += [ nn.GELU() ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True) ]
        ly += [ nn.GELU() ]

        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True) ]
        ly += [ nn.GELU() ]
        
        self.body = nn.Sequential(*ly)

    def forward(self, x): # x:[B,D,H,W]
        return self.body(x)

class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride, bias=True) ]
        ly += [ nn.GELU() ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)

class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)