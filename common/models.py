"""Model registry. Every model maps a [B,3,H,W] input to an oil probability map [B,1,H,W] (sigmoid).

M1 cbdnet      official CBD-Net code (third_party/CBDNet-main), unchanged.
M2 dlv3p_mnv2  Improved DeepLabV3+ [P01], re-implemented from the paper description:
               MobileNetV2 (ImageNet) backbone with an scSE module at the input of every
               inverted-residual block; ASPP (1x1, atrous 6/12/18, image pooling) with scSE on the
               concatenated branches; DeepLabV3+ decoder with the stride-4 skip; output stride 16.
               Assumed where the paper is silent: cSE reduction 16, 256 ASPP channels,
               48-channel low-level projection, standard (non-separable) 3x3 convolutions.
M3 fcsnet      FCS-Net [P29], re-implemented from the module outline: see fcsnet.py (assumptions marked there).
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "third_party", "CBDNet-main"))
os.environ.setdefault("TORCH_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "torch"))


# ---------------------------------------------------------------- M2 building blocks
class SCSE(nn.Module):
    """Concurrent spatial and channel squeeze-and-excitation (Roy et al. 2018), additive."""

    def __init__(self, ch, r=16):
        super().__init__()
        hid = max(ch // r, 4)
        self.cse = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, hid, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(hid, ch, 1), nn.Sigmoid())
        self.sse = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)


class SCSEThen(nn.Module):
    """scSE applied to the input of a backbone block, then the block."""

    def __init__(self, block, in_ch):
        super().__init__()
        self.att, self.block = SCSE(in_ch), block

    def forward(self, x):
        return self.block(self.att(x))


def _cbr(i, o, k=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=0 if k == 1 else d, dilation=d, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ASPP_SCSE(nn.Module):
    def __init__(self, in_ch, ch=256, rates=(6, 12, 18)):
        super().__init__()
        self.b = nn.ModuleList([_cbr(in_ch, ch, 1)] + [_cbr(in_ch, ch, 3, r) for r in rates])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, ch, 1, bias=False),
                                  nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.att = SCSE(ch * (len(rates) + 2))
        self.proj = nn.Sequential(_cbr(ch * (len(rates) + 2), ch, 1), nn.Dropout(0.1))

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.b]
        outs.append(F.interpolate(self.pool(x), size=(h, w), mode="bilinear", align_corners=False))
        return self.proj(self.att(torch.cat(outs, 1)))


class DeepLabV3PlusMNV2(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        feats = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None).features
        # output stride 16: remove the stride of block 14 and dilate blocks 14-17
        for i in range(14, 18):
            for m in feats[i].modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                    if i == 14:
                        m.stride = (1, 1)
                    m.dilation, m.padding = (2, 2), (2, 2)
        blocks = []
        for i in range(18):   # drop features[18] (1x1 conv to 1280)
            b = feats[i]
            if i >= 1:        # inverted-residual blocks 1..17
                in_ch = b.conv[0][0].in_channels if i > 1 else 32
                b = SCSEThen(b, in_ch)
            blocks.append(b)
        self.low = nn.Sequential(*blocks[:4])     # stride 4, 24 channels
        self.high = nn.Sequential(*blocks[4:])    # stride 16, 320 channels
        self.aspp = ASPP_SCSE(320)
        self.low_proj = _cbr(24, 48, 1)
        self.dec = nn.Sequential(_cbr(256 + 48, 256, 3), _cbr(256, 256, 3), nn.Conv2d(256, 1, 1))

    def forward(self, x):
        H, W = x.shape[-2:]
        lo = self.low(x)
        hi = self.aspp(self.high(lo))
        hi = F.interpolate(hi, size=lo.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec(torch.cat([hi, self.low_proj(lo)], 1))
        return torch.sigmoid(F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False))


# ---------------------------------------------------------------- registry
def build_model(arch, pretrained=True):
    if arch == "cbdnet":
        from networks.CBDNet import CBDNet  # official
        return CBDNet()
    if arch == "dlv3p_mnv2":
        return DeepLabV3PlusMNV2(pretrained=pretrained)
    if arch == "fcsnet":
        from fcsnet import FCSNet
        return FCSNet(pretrained=pretrained)
    raise ValueError(f"unknown arch {arch}")


def complexity(net, size=256):
    """Parameters (M) and GMACs for one 3x256x256 input (FLOP counter halves to MACs)."""
    from torch.utils.flop_counter import FlopCounterMode
    params = sum(p.numel() for p in net.parameters()) / 1e6
    net = net.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        net(torch.zeros(1, 3, size, size, device=next(net.parameters()).device))
    return params, fc.get_total_flops() / 2 / 1e9
