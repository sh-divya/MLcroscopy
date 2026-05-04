import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from mlcroscopy.models.blocks import (
    conv_block,
    linear_block,
    res_block,
    unfreeze_except_lastN,
)

from omegaconf import DictConfig


class ScopyRes(nn.Module):

    def __init__(
        self,
        micro_kwargs,
        xrf_kwargs,
        merge,
        expt_kwargs,
        pred_head_kwargs,
        batch_norm=True,
        droprate=False,
    ):
        super().__init__()
        self.act = nn.ReLU

        if "name" in micro_kwargs:
            name = micro_kwargs["name"]
            name = f"facebook/{name}"
            self.micro_block = AutoModel.from_pretrained(name, device_map="auto")
            self.proc = AutoImageProcessor.from_pretrained(name)

            frozen_layers = micro_kwargs.get("unfrozen_layers", 0)
            unfreeze_except_lastN(self.micro_block, frozen_layers)
            # for param in self.micro_block.parameters():
            #     param.requires_grad = False

            self.model_hidden_len = micro_kwargs.get("hidden", 384)
            self.pretrain = True
        else:
            self.micro_block = res_block(
                3, micro_kwargs["kernel"], micro_kwargs["layers"], batch_norm, droprate
            )
            self.micro_block = nn.Sequential(*self.micro_block)
            self.pretrain = False

        self.xrf_block = res_block(
            4, xrf_kwargs["kernel"], xrf_kwargs["layers"], batch_norm, droprate
        )
        self.xrf_block = nn.Sequential(*self.xrf_block)

        if merge == "flatten":
            self.merge = [nn.Identity()]
            self.model_hidden_len = (
                micro_kwargs["hidden"]
                if self.pretrain
                else (micro_kwargs["final_dim"] ** 2) * micro_kwargs["layers"][-1]
            )
            self.final_dim = (
                self.model_hidden_len
                + (xrf_kwargs["final_dim"] ** 2) * xrf_kwargs["layers"][-1]
            )
        elif merge == "global":
            self.merge = [nn.AdaptiveAvgPool2d(1)]
            self.model_hidden_len = (
                micro_kwargs["hidden"] if self.pretrain else micro_kwargs["layers"][-1]
            )
            self.final_dim = self.model_hidden_len + xrf_kwargs["layers"][-1]
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

        if self.pretrain:
            micro_h = self.proc(micro_h, return_tensors="pt", do_rescale=True)[
                "pixel_values"
            ]
            with torch.inference_mode():
                micro_h = self.micro_block(micro_h).last_hidden_state[:, 0, :]
        else:
            micro_h = self.micro_block(micro_h)
            micro_h = self.merge(micro_h)

        xrf_h = self.xrf_block(xrf_h)
        xrf_h = self.merge(xrf_h)

        h = torch.concat([micro_h, xrf_h], 1)
        if self.expt_kwargs["do"]:
            h = torch.concat([h, expt_h], 1)

        h = self.pred_block(h)

        return self.final_layer(h)


class ScopyDino(nn.Module):

    def __init__(
        self,
        foundation_model_kwargs,
        xrf_kwargs,
        merge,
        expt_kwargs,
        pred_head_kwargs,
        batch_norm=True,
        droprate=0.0,
    ):
        super().__init__()
        name = foundation_model_kwargs.get("name", "dinov3-vits16-pretrain-lvd1689m")
        self.name = f"facebook/{name}"
        self.model = AutoModel.from_pretrained(self.name, device_map="auto")
        self.pretrain = True
        self.embedding_type = foundation_model_kwargs.get("embedding_type", "cls")
        self.proc = AutoImageProcessor.from_pretrained(self.name)
        self.model_hidden_len = foundation_model_kwargs.get("hidden", 384)

        frozen_layers = foundation_model_kwargs.get("unfrozen_layers", 0)
        unfreeze_except_lastN(self.model, frozen_layers)
        # for param in self.micro_block.parameters():
        #     param.requires_grad = False

        self.act = nn.ReLU

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
            self.final_dim = (
                self.model_hidden_len
                + (xrf_kwargs["final_dim"] ** 2) * xrf_kwargs["layers"][-1]
            )
        elif merge == "global":
            self.merge = [nn.AdaptiveAvgPool2d(1)]
            self.final_dim = self.model_hidden_len + xrf_kwargs["layers"][-1]
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

        micro_h = self.proc(micro_h, return_tensors="pt", do_rescale=True)[
            "pixel_values"
        ]
        with torch.inference_mode():
            if self.embedding_type == "cls":
                micro_h = self.model(micro_h).last_hidden_state[:, 0, :]
            elif self.embedding_type == "mean":
                micro_h = self.model(micro_h).last_hidden_state[:, 5::]
                micro_h = micro_h.mean(dim=1)
            elif self.embedding_type == "concat":
                cls_feat = self.model(micro_h).last_hidden_state[:, 0, :]
                mean_patch = self.model(micro_h).last_hidden_state[:, 5::]
                mean_patch = mean_patch.mean(dim=1)
                micro_h = torch.cat([cls_feat, mean_patch], dim=-1)
            else:
                raise KeyError(
                    f"{self.embedding_type} is not available as feature extraction for dinov3"
                )

        xrf_h = self.xrf_block(xrf_h)
        xrf_h = self.merge(xrf_h)

        h = torch.concat([micro_h, xrf_h], 1)
        if self.expt_kwargs["do"]:
            h = torch.concat([h, expt_h], 1)

        h = self.pred_block(h)

        return self.final_layer(h)


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
