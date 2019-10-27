import torch.nn as nn
import torch.nn.functional as F

class ZeroCenter(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        """x : [B, C, H, W]"""
        return x.sub_(x.flatten(1).mean(1, keepdim=True).unsqueeze(-1).unsqueeze(-1))

EPS = 1e-5

class ZeroNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        """x : [B, C, H, W]"""
        """x_mean : [B, 1, 1, 1]"""
        return F.layer_norm(x, x.size()[1:], None, None, EPS)

class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, center='before', **batchnorm_params):

        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size,
                stride=stride, 
                padding=padding, 
                bias=not (use_batchnorm)
            )
        ]

        if use_batchnorm == 'inplace':
            try:
                from inplace_abn import InPlaceABN
            except ImportError:
                raise RuntimeError("In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. To install see: https://github.com/mapillary/inplace_abn")
            
            layers.append(InPlaceABN(out_channels, activation='leaky_relu', activation_param=0.0, **batchnorm_params))
        elif use_batchnorm:
            if center == 'before':
                layers.append(ZeroCenter())
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.BatchNorm2d(out_channels, **batchnorm_params))
            elif center == 'after':
                layers.append(nn.BatchNorm2d(out_channels, **batchnorm_params))
                layers.append(nn.ReLU(inplace=False))
                layers.append(ZeroCenter())
            elif center == 'norm':
                layers.append(nn.BatchNorm2d(out_channels, **batchnorm_params))
                layers.append(nn.ReLU(inplace=True))
                layers.append(ZeroNorm())
            else:
                layers.append(nn.BatchNorm2d(out_channels, **batchnorm_params))
                layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
                
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch, ch//re, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//re, ch, 1),
                                 nn.Sigmoid()
                                )
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
