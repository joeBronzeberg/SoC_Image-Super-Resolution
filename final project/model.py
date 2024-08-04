import torch
from torch import nn
from math import sqrt

class FSRCNN(nn.Module):
    
    def __init__(self, scale_factor, c=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()

        # Feature extraction layer.
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(c, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )

        # Shrinking layer.
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )

        # Mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(s, s, kernel_size=3, padding=3//2),
            nn.PReLU(s)
        )
        for _ in range(m-1):
            self.map.add_module(f'conv_{_}', nn.Conv2d(s, s, kernel_size=3, padding=3//2))
            self.map.add_module(f'prelu_{_}', nn.PReLU(s))

        # Expanding layer.
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )

        # Deconvolution layer.
        self.deconv = nn.ConvTranspose2d(d, c, kernel_size=9, stride=scale_factor, padding=9//2, output_padding=scale_factor-1)

        self._initialize_weights()

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        x = self.deconv(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

