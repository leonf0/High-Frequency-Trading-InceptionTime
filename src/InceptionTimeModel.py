import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels=40, out_channels=32, bottleneck_channels=32, kernels=(3, 7, 15)):
        super().__init__()

        self.use_bottleneck = in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            conv_in = bottleneck_channels
        else:
            conv_in = in_channels

        self.branches = nn.ModuleList([
            nn.Conv1d(conv_in, out_channels, k, padding=k // 2, bias=False)
            for k in kernels
        ])

        self.pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        n_branches = len(kernels) + 1
        self.bn = nn.BatchNorm1d(out_channels * n_branches)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        if self.use_bottleneck:
            x_bn = self.bottleneck(x)
        else:
            x_bn = x

        outs = [branch(x_bn) for branch in self.branches]
        outs.append(self.pool_conv(self.pool(x)))
        out = torch.cat(outs, dim=1)
        return self.dropout(self.act(self.bn(out)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=40, out_channels=32, bottleneck_channels=32, kernels=(3, 7, 15)):
        super().__init__()
        n_filters = out_channels * (len(kernels) + 1)

        self.inception1 = InceptionBlock(in_channels, out_channels, bottleneck_channels, kernels)
        self.inception2 = InceptionBlock(n_filters, out_channels, bottleneck_channels, kernels)
        self.inception3 = InceptionBlock(n_filters, out_channels, bottleneck_channels, kernels)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(n_filters),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.inception1(x)
        out = self.inception2(out)
        out = self.inception3(out)
        return self.act(out + self.shortcut(x))


class InceptionTime(nn.Module):
    def __init__(self, in_channels=40, n_classes=3, n_blocks=2, out_channels=32,
                 bottleneck_channels=32, kernels=(3, 7, 15)):
        super().__init__()
        n_filters = out_channels * (len(kernels) + 1)

        blocks = []
        for i in range(n_blocks):
            block_in = in_channels if i == 0 else n_filters
            blocks.append(ResidualBlock(block_in, out_channels, bottleneck_channels, kernels))
        self.blocks = nn.Sequential(*blocks)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head_dropout = nn.Dropout(0.35)
        self.head = nn.Linear(n_filters, n_classes)

    def forward(self, x):
        out = self.blocks(x)
        out = self.gap(out).squeeze(-1)
        out = self.head_dropout(out)
        return self.head(out)
