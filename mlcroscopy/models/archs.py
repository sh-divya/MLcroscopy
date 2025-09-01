import torch
import torch.nn as nn


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
    def __init__(
        self, res_type, res_layers, res_width, droprate
    ):
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
                self.act
            ]
        )

    def forward(self, x):

        x = self.pool1(self.init_conv(x))
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
        x = self.pool(self.init_conv(x))
        if self.bn1:
            x = self.bn1(x)
        x = self.act(x)
        for l in self.conv2:
            x = l(x)
        return x



class ScopyResNet(nn.Module):
    def __init__(
        self,
        pred_head_len,
        pred_head_width,
        pred_hidden_len,
        micro_inp_width,
        micro_out_width,
        micro_res_n,
        xrf_conv_width,
        xrf_res_len,
        batch_norm,
        droprate=None,
    ):
        super().__init__()
        self.micro_inp_width = micro_inp_width
        self.micro = MicroBlock("basic", micro_res_n, micro_out_width, droprate)
        self.xrf = XRFBlock(xrf_conv_width, xrf_res_len, batch_norm)

        self.final_width = xrf_conv_width + micro_out_width
        self.conv = nn.Conv2d(self.final_width, pred_head_width, 3, 1, 1)
        if droprate:
            self.wrn = nn.ModuleList([WideResBlock(pred_head_width, droprate) for _ in range(pred_head_len)])
        else:
            self.wrn = nn.ModuleList([WideResBlock(pred_head_width, 0.0) for _ in range(pred_head_len)])
        self.bn = nn.BatchNorm2d(pred_head_width) if batch_norm else None
        self.drop = nn.Dropout(droprate)
        self.act = nn.ReLU()
        self.fc = []
        for l in range(pred_hidden_len):
            if l == 0:
                layer = nn.Linear(24 * 24 * pred_head_width, pred_head_width)
            else:
                layer = nn.Linear(pred_head_width, pred_head_width)
            self.fc.append(nn.Dropout(droprate))
            self.fc.append(layer)
            if l < pred_head_len - 1:
                if batch_norm:
                    self.fc.append(nn.BatchNorm1d(pred_head_width))
                self.fc.append(nn.ReLU())
        self.fc.append(nn.Linear(pred_head_width, 2))
        self.fc = nn.ModuleList(self.fc)
        self.final_act = nn.ReLU()

    def forward(self, x):
        micro_x, xrf_x = x
        # print(micro_x.shape)
        micro_x = self.micro(micro_x)
        xrf_x = self.xrf(xrf_x)
        # print(micro_x.shape)
        # print(xrf_x.shape)
        x = self.conv(torch.concat([micro_x, xrf_x], 1))
        for l in self.wrn:
            x = l(x)
        if self.bn:
            x = self.bn(x)
        x = self.drop(self.act(x))
        x = torch.flatten(x, 1)
        for l in self.fc:
            x = l(x)
        x = self.final_act(x)

        return x
    
class ExptMLP(nn.Module):
    def __init__(self, hidden_dim, hidden_layers, batch_norm=False):
        super().__init__()
        self.layers = []
        for h in range(hidden_layers):
            if h == 0:
                l = nn.Linear(5, hidden_dim)
            else:
                l = nn.Linear(hidden_dim, hidden_dim)
            self.layers.append(l)
            if h < hidden_layers - 1:
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(hidden_dim))
                self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, 2))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
