import torch
import torch.nn as nn

from omegaconf import DictConfig

class WideResBlock(nn.Module):
    def __init__(self, width, droprate=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(width)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(width, width, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(width)
        self.act2 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(droprate)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1)

    def forward(self, x):
        out = self.act1(self.bn1(x))
        out = self.act2(self.bn2(self.conv1(out)))
        out = self.conv2(self.drop(out))
        return torch.add(x, out)


class MicroBlock(nn.Module):
    def __init__(self, res_type, res_layers, res_width, droprate):
        super().__init__()
        if res_type == "basic":
            res_type = WideResBlock
        elif res_type == "unequal":
            raise NotImplementedError
        else:
            raise ValueError(f"{res_type} residual block not available")

        self.init_conv = nn.Conv2d(3, res_width, 7, 2, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.res_blocks = []
        # if batch_norm:
        #     batch_norm = nn.BatchNorm2d(res_width)
        # else:
        #     batch_norm = None

        self.act = nn.ReLU()

        for l in range(res_layers):
            layer = res_type(res_width, droprate)
            self.res_blocks.append(layer)

        self.res_blocks = nn.ModuleList(self.res_blocks)
        self.after_blocks = nn.ModuleList(
            [
                nn.Conv2d(res_width, res_width, 3, 1, 1),
                nn.BatchNorm2d(res_width),
                self.act,
                nn.MaxPool2d(2, 2),
                nn.Conv2d(res_width, res_width, 7, 1, 0),
                nn.BatchNorm2d(res_width),
                self.act,
            ]
        )

    def forward(self, x):

        x = self.init_conv(x)
        x = self.act(self.pool1(x))
        for l in self.res_blocks:
            x = l(x)

        # if self.batch_norm:
        #     x = self.batch_norm(x)

        for l in self.after_blocks:
            x = l(x)

        return x


class XRFBlock(nn.Module):
    def __init__(self, conv_width, conv_res_len, batch_norm=False):
        super().__init__()
        self.init_conv = nn.Conv2d(4, conv_width, 4, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(conv_width) if batch_norm else False
        self.act = nn.ReLU(inplace=True)

        self.conv2 = []
        for l in range(conv_res_len - 1):
            self.conv2.append(nn.Conv2d(conv_width, conv_width, 3, 1, 1))
            if batch_norm:
                self.conv2.append(nn.BatchNorm2d(conv_width))
            self.conv2.append(nn.ReLU())

        self.conv2 = nn.ModuleList(self.conv2)
        self.res_block = conv_res_len

    def forward(self, x):
        x = self.init_conv(x)
        x = self.act(self.pool(x))
        if self.bn1:
            x = self.bn1(x)
        x = self.act(x)
        for l in self.conv2:
            x = l(x)
        return x

def res_block():
    pass


def conv_block(
    input_channels, kernel, layers, downsample, activation, batch_norm, droprate
):
    block = []
    curr = input_channels

    for i, l in enumerate(layers):
        block.append(nn.Conv2d(curr, l, **kernel))
        curr = l
        if batch_norm:
            block.append(nn.BatchNorm2d(l))
        block.append(activation())
        block.append(nn.Dropout(droprate if droprate else 0.0))
        if downsample is not None:
            if downsample["type"] == "conv":
                block.append(nn.Conv2d(curr, curr, **downsample["kernel"]))
            elif downsample["type"] == "pool":
                block.append(nn.MaxPool2d(**downsample["kernel"]))
            elif downsample["type"] == "skip":
                pass
            else:
                raise KeyError("Downsample type has not been implemented")

    return block


def linear_block(input_dim, layers, activation, batch_norm, droprate):
    block = []
    curr = input_dim

    if isinstance(layers, (dict, DictConfig)):
        layers = [layers["hidden"] for _ in range(layers["num"])]


    for i, l in enumerate(layers):
        block.append(nn.Linear(curr, l))
        curr = l
        if batch_norm:
            block.append(nn.BatchNorm1d(l))
        block.append(activation())
        block.append(nn.Dropout(droprate if droprate else 0.0))

    return block
