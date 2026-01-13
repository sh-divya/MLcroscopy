import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from mlcroscopy.models.blocks import WideResBlock, XRFBlock, MicroBlock


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
            self.wrn = nn.ModuleList(
                [WideResBlock(pred_head_width, droprate) for _ in range(pred_head_len)]
            )
        else:
            self.wrn = nn.ModuleList(
                [WideResBlock(pred_head_width, 0.0) for _ in range(pred_head_len)]
            )
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
            if l < pred_hidden_len - 1:
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
        # x = self.final_act(x)

        return x


class ScopyResLight(nn.Module):
    def __init__(
        self,
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
        curr_width = self.final_width

        self.fc = []
        for l in range(pred_hidden_len):
            if l == 0:
                layer = nn.Linear(24 * 24 * curr_width, pred_head_width)
            else:
                layer = nn.Linear(pred_head_width, pred_head_width)
            self.fc.append(nn.Dropout(droprate))
            self.fc.append(layer)
            if l < pred_hidden_len - 1:
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
        x = torch.concat([micro_x, xrf_x], 1)

        x = torch.flatten(x, 1)
        for l in self.fc:
            x = l(x)
        # x = self.final_act(x)

        return x


class ScopyDino(nn.Module):
    def __init__(
        self,
        pred_hidden_len,
        pred_head_width,
        xrf_conv_width,
        xrf_res_len,
        droprate=0.0,
        batch_norm=True,
        name="dinov3-vits16-pretrain-lvd1689m",
    ):
        super().__init__()
        name = f"facebook/{name}"
        self.model = AutoModel.from_pretrained(name, device_map="auto")
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model_hidden_len = 384

        for param in self.model.parameters():
            param.requires_grad = False

        self.xrf = XRFBlock(xrf_conv_width, xrf_res_len, batch_norm)
        self.xrf_flatten_len = 24 * 24 * xrf_conv_width
        curr_width = self.model_hidden_len + self.xrf_flatten_len

        self.fc = []
        for l in range(pred_hidden_len):
            if l == 0:
                layer = nn.Linear(curr_width, pred_head_width)
            else:
                layer = nn.Linear(pred_head_width, pred_head_width)
            self.fc.append(nn.Dropout(droprate))
            self.fc.append(layer)
            if l < pred_hidden_len - 1:
                if batch_norm:
                    self.fc.append(nn.BatchNorm1d(pred_head_width))
                self.fc.append(nn.ReLU())
        self.fc.append(nn.Linear(pred_head_width, 2))
        self.fc = nn.ModuleList(self.fc)
        self.final_act = nn.ReLU()

    def forward(self, x):
        micro_x, xrf_x = x

        micro_x = self.proc(micro_x, return_tensors="pt", do_rescale=True)[
            "pixel_values"
        ]
        with torch.inference_mode():
            micro_h = self.model(micro_x).last_hidden_state[:, 0, :]

        xrf_h = self.xrf(xrf_x)

        xrf_h = torch.flatten(xrf_h, 1)
        h = torch.concat([micro_h, xrf_h], 1)

        for l in self.fc:
            h = l(h)
        return h


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


class BothTensorandExpt(nn.Module):
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
        expt_hidden_len,
        batch_norm,
        droprate=None,
    ):
        super().__init__()
        self.micro_inp_width = micro_inp_width
        self.micro = MicroBlock("basic", micro_res_n, micro_out_width, droprate)
        self.xrf = XRFBlock(xrf_conv_width, xrf_res_len, batch_norm)

        self.final_width = xrf_conv_width + micro_out_width
        curr_width = self.final_width

        self.expt_mlp = []
        for h in range(expt_hidden_len):
            if h == 0:
                l = nn.Linear(5, pred_head_width)
            else:
                l = nn.Linear(pred_head_width, pred_head_width)
            self.expt_mlp.append(l)
            if h < expt_hidden_len - 1:
                if batch_norm:
                    self.expt_mlp.append(nn.BatchNorm1d(pred_head_width))
                self.expt_mlp.append(nn.ReLU())
        self.expt_mlp = nn.ModuleList(self.expt_mlp)

        self.fc = []
        for l in range(pred_hidden_len):
            if l == 0:
                layer = nn.Linear(
                    24 * 24 * curr_width + pred_head_width, pred_head_width
                )
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
        micro_x, xrf_x, expt_x = x
        micro_x = self.micro(micro_x)
        xrf_x = self.xrf(xrf_x)
        for l in self.expt_mlp:
            expt_x = l(expt_x)

        micro_x = torch.flatten(micro_x, 1)
        xrf_x = torch.flatten(xrf_x, 1)
        x = torch.concat([micro_x, xrf_x, expt_x], 1)

        for l in self.fc:
            x = l(x)
        x = self.final_act(x)

        return x

class ScopyConvBasic(nn.Module):
    pass


class ScopyAllConv(nn.Module):
    pass