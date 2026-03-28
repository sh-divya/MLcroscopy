import torch
import torch.nn as nn

from omegaconf import DictConfig


class LightRes(nn.Module):
    def __init__(
        self, inp_channels, out_channels, kernel_kwargs, batch_norm=True, droprate=0.0
    ):
        super().__init__()
        self.act = nn.ReLU()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.stride = kernel_kwargs["stride"]
        self.drop = nn.Dropout(droprate)
        self.downsample = nn.AvgPool2d(2, 2)

        self.conv1 = nn.Conv2d(
            inp_channels,
            out_channels,
            kernel_kwargs["kernel_size"],
            self.stride,
            kernel_kwargs["padding"],
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            1,
            1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.residual = nn.Sequential(
            self.conv1, self.bn1, self.act, self.drop, self.conv2, self.bn2
        )

    def skip_connection(self, x):
        if self.stride != 1 or self.inp_channels != self.out_channels:
            # x = x[:, :, :: self.stride, :: self.stride]
            x = self.downsample(x)
            if self.inp_channels < self.out_channels:
                pad = self.out_channels - self.inp_channels
                x = nn.functional.pad(
                    x, (0, 0, 0, 0, 0, pad), mode="constant", value=0.0
                )
                # print("Padded OG", x.shape)
            else:
                x = x[:, : self.out_channels, :, :]
                # print("Sliced OG", x.shape)

        return x

    def forward(self, x):
        # print("Input", x.shape)
        residual = self.residual(x)
        # print("Residual", residual.shape)
        skip = self.skip_connection(x)

        out = residual + skip
        return self.act(out)


def res_block(input_channels, kernel, layers, batch_norm, droprate):

    block = []
    curr = input_channels

    for i, l in enumerate(layers):
        block.append(LightRes(curr, l, kernel, batch_norm, droprate))
        curr = l

    return block


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
