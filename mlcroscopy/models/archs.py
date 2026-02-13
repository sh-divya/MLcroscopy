import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from mlcroscopy.models.blocks import (
    WideResBlock,
    XRFBlock,
    MicroBlock,
    conv_block,
    linear_block,
)

from omegaconf import DictConfig


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

        micro_x = self.micro(micro_x)
        xrf_x = self.xrf(xrf_x)
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

        micro_x = self.micro(micro_x)
        xrf_x = self.xrf(xrf_x)
        x = torch.concat([micro_x, xrf_x], 1)

        x = torch.flatten(x, 1)
        for l in self.fc:
            x = l(x)
        # x = self.final_act(x)

        return x


# dino_model="dinov3-vits16-pretrain-lvd1689m",
# xrf_kwargs,
# pred_head_kwargs,
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
    def __init__(self, hidden_kwargs, batch_norm=False, droprate=0.0):
        super().__init__()
        self.layers = []
        self.act = nn.ReLU

        self.layers = linear_block(5, hidden_kwargs, self.act, batch_norm, droprate)

        if isinstance(hidden_kwargs, (dict, DictConfig)):
            self.final_hidden = hidden_kwargs["hidden"]
        else:
            self.final_hidden = hidden_kwargs[-1]
        self.layers.append(nn.Linear(self.final_hidden, 2))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


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


class ScopyClassic(nn.Module):
    def __init__(
        self,
        micro_kwargs,
        xrf_kwargs,
        expt_kwargs,
        merge,
        pred_head_kwargs,
        batch_norm,
        droprate,
    ):
        super().__init__()
        self.act = nn.ReLU

        self.micro_block = conv_block(
            3,
            micro_kwargs["kernel"],
            micro_kwargs["layers"],
            micro_kwargs["down"],
            self.act,
            batch_norm,
            droprate,
        )
        self.micro_block = nn.Sequential(*self.micro_block)
        # self.micro_block = nn.ModuleList(self.micro_block)
        self.xrf_block = conv_block(
            4,
            xrf_kwargs["kernel"],
            xrf_kwargs["layers"],
            xrf_kwargs["down"],
            self.act,
            batch_norm,
            droprate,
        )
        self.xrf_block = nn.Sequential(*self.xrf_block)

        if merge == "flatten":
            self.merge = [nn.Identity()]
            self.final_dim = (micro_kwargs["final_dim"] ** 2) * micro_kwargs["layers"][
                -1
            ] + (xrf_kwargs["final_dim"] ** 2) * xrf_kwargs["layers"][-1]
        elif merge == "global":
            self.merge = [nn.AdaptiveAvgPool2d(1)]
            self.final_dim = micro_kwargs["layers"][-1] + xrf_kwargs["layers"][-1]
        else:
            raise KeyError("Merge technique hasn't been specified")
        self.merge.append(nn.Flatten())
        self.merge = nn.Sequential(*self.merge)

        self.expt_kwargs = expt_kwargs
        if self.expt_kwargs["do"]:
            self.expt_block = linear_block(
                5, self.expt_kwargs["layers"], self.act, batch_norm, droprate
            )
            self.expt_block = nn.Sequential(*self.expt_block)
            if isinstance(self.expt_kwargs["layers"], (dict, DictConfig)):
                hidden = self.expt_kwargs["layers"]["hidden"]
            else:
                hidden = self.expt_kwargs["layers"][-1]
            self.final_dim = self.final_dim + hidden
        else:
            pass

        if isinstance(pred_head_kwargs, (dict, DictConfig)):
            self.final_hidden = pred_head_kwargs["hidden"]
        else:
            self.final_hidden = pred_head_kwargs[-1]
        self.pred_block = linear_block(
            self.final_dim, pred_head_kwargs, self.act, batch_norm, droprate
        )
        self.pred_block = nn.Sequential(*self.pred_block)
        self.final_layer = nn.Linear(self.final_hidden, 2)

    def forward(self, x):
        if self.expt_kwargs["do"]:
            micro_h, xrf_h, expt_h = x
            expt_h = self.expt_block(expt_h)
        else:
            micro_h, xrf_h = x

        micro_h = self.micro_block(micro_h)
        micro_h = self.merge(micro_h)

        xrf_h = self.xrf_block(xrf_h)
        xrf_h = self.merge(xrf_h)

        h = torch.concat([micro_h, xrf_h], 1)
        if self.expt_kwargs["do"]:
            h = torch.concat([h, expt_h], 1)

        h = self.pred_block(h)

        return self.final_layer(h)
