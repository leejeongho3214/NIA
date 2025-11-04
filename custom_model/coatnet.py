import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange


class DropPath(nn.Module):
    """Stochastic depth per sample when applied in residual branches."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=True),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MBConv(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        image_size,
        downsample=False,
        expansion=4,
        drop_path=0.0,
    ):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=True)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=True),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=True),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=True),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=True),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=True),
                nn.BatchNorm2d(oup),
            )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = self.drop_path(self.conv(x))
        if self.downsample:
            shortcut = self.proj(self.pool(x))
        else:
            shortcut = x
        return shortcut + residual


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)), indexing='ij')
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=True) 

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        image_size,
        heads=8,
        dim_head=32,
        downsample=False,
        dropout=0.,
        drop_path=0.0,
    ):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=True)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),      # LayerLorm based on feature dim, not batch dim like BatchNorm
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        if self.downsample:
            shortcut = self.proj(self.pool1(x))
            x = shortcut + self.drop_path(self.attn(self.pool2(x)))
        else:
            x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.ff(x))
        return x


class CoAtNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        num_blocks,
        channels,
        num_classes=1000,
        block_types=['C', 'C', 'T', 'T'],
        dropout=0.0,
        drop_path_rate=0.0,
        classifier_dropout=0.0,
    ):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        stochastic_depth_blocks = sum(num_blocks[1:])
        if drop_path_rate > 0.0 and stochastic_depth_blocks > 0:
            drop_path_rates = torch.linspace(
                0, drop_path_rate, stochastic_depth_blocks
            ).tolist()
        else:
            drop_path_rates = [0.0] * stochastic_depth_blocks

        rate_offset = 0

        def take_rates(n, enable=True):
            nonlocal rate_offset
            if not enable or n == 0:
                return [0.0] * n
            rates = drop_path_rates[rate_offset:rate_offset + n]
            rate_offset += n
            return rates

        self.s0 = self._make_layer(
            conv_3x3_bn,
            in_channels,
            channels[0],
            num_blocks[0],
            (ih // 2, iw // 2),
            drop_path_rates=take_rates(num_blocks[0], enable=False),
            dropout=0.0,
        )
        self.s1 = self._make_layer(
            block[block_types[0]],
            channels[0],
            channels[1],
            num_blocks[1],
            (ih // 4, iw // 4),
            drop_path_rates=take_rates(num_blocks[1]),
            dropout=dropout,
        )
        self.s2 = self._make_layer(
            block[block_types[1]],
            channels[1],
            channels[2],
            num_blocks[2],
            (ih // 8, iw // 8),
            drop_path_rates=take_rates(num_blocks[2]),
            dropout=dropout,
        )
        self.s3 = self._make_layer(
            block[block_types[2]],
            channels[2],
            channels[3],
            num_blocks[3],
            (ih // 16, iw // 16),
            drop_path_rates=take_rates(num_blocks[3]),
            dropout=dropout,
        )
        self.s4 = self._make_layer(
            block[block_types[3]],
            channels[3],
            channels[4],
            num_blocks[4],
            (ih // 32, iw // 32),
            drop_path_rates=take_rates(num_blocks[4]),
            dropout=dropout,
        )

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.head_dropout = nn.Dropout(classifier_dropout) if classifier_dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(channels[-1], num_classes, bias=True)   

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = torch.flatten(self.pool(x), 1)
        x = self.head_dropout(x)
        x = self.fc(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size, drop_path_rates=None, dropout=0.0):
        layers = nn.ModuleList([])
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        for i in range(depth):
            in_channels = inp if i == 0 else oup
            downsample = i == 0
            drop_rate = drop_path_rates[i] if i < len(drop_path_rates) else drop_path_rates[-1]

            if block is MBConv:
                layers.append(
                    block(
                        in_channels,
                        oup,
                        image_size,
                        downsample=downsample,
                        drop_path=drop_rate,
                    )
                )
            elif block is Transformer:
                layers.append(
                    block(
                        in_channels,
                        oup,
                        image_size,
                        downsample=downsample,
                        dropout=dropout,
                        drop_path=drop_rate,
                    )
                )
            else:
                layers.append(block(in_channels, oup, image_size, downsample=downsample))
        return nn.Sequential(*layers)


def coatnet_0():
    num_blocks = [2, 2, 3, 5, 2]            # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_1():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_2():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [128, 128, 256, 512, 1026]   # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_3():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_4(num_classes=1000, drop_path_rate=0.2, dropout=0.1, classifier_dropout=0.1):
    num_blocks = [2, 2, 12, 28, 2]          # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet(
        (256, 256),
        3,
        num_blocks,
        channels,
        num_classes,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        classifier_dropout=classifier_dropout,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)

    net = coatnet_0()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_1()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_2()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_3()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_4()
    out = net(img)
    print(out.shape, count_parameters(net))
