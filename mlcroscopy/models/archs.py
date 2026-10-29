from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from transformers import AutoModel, AutoImageProcessor
from mlcroscopy.models.blocks import (
    conv_block,
    linear_block,
    res_block,
    unfreeze_except_lastN,
    zeros_like_embedding,
)

from omegaconf import DictConfig


class ScopyNASA(nn.Module):

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
        self.act = nn.ReLU

        self.micro_use = foundation_model_kwargs.get("use", "train")
        self.xrf_use = xrf_kwargs.get("use", "train")

        name = foundation_model_kwargs.get("name", "resnet50")
        url_base = (
            "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/"
        )
        url_end = "_v1.0.pth.tar"
        self.name = name

        self.micro_embed = foundation_model_kwargs.get("use", "train")
        if self.micro_use in ["train", "zeros"]:
            url = url_base + f'{name}_pretrained_{"microscopynet"}' + url_end
            self.model_hidden_len = foundation_model_kwargs.get("hidden", 2048)
            self.model = torch.hub.load(
                "pytorch/vision:v0.10.0", name, pretrained=False
            )
            self.model.load_state_dict(model_zoo.load_url(url))
            self.model.fc = nn.Identity()
            self.pretrain = True
            self.frozen_layers = foundation_model_kwargs.get("unfrozen_layers", 0)
            unfreeze_except_lastN(self.model, self.frozen_layers)
            self.model.eval()
        elif self.micro_use in ["skip"]:
            self.model_hidden_len = 0

        if self.xrf_use in ["train", "zeros"]:
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
            xrf_dim_mult = 1
        elif self.xrf_use in ["skip"]:
            xrf_dim_mult = 0

        if merge == "flatten":
            self.merge = [nn.Identity()]
            self.xrf_dim = +(xrf_kwargs["final_dim"] ** 2) * xrf_kwargs["layers"][-1]
        elif merge == "global":
            self.merge = [nn.AdaptiveAvgPool2d(1)]
            self.xrf_dim = xrf_kwargs["layers"][-1]
        else:
            raise KeyError("Merge technique hasn't been specified")

        self.xrf_dim = self.xrf_dim * xrf_dim_mult
        self.final_dim = self.model_hidden_len + self.xrf_dim
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

    def get_micro_embed(self, micro_x):
        if self.micro_use == "train":
            micro_h = self.model(micro_x)
        elif self.micro_use == "zeros":
            micro_h = zeros_like_embedding(micro_x, self.model_hidden_len)
        elif self.micro_use == "skip":
            micro_h = None
        else:
            raise ValueError(f"Unknown micro use mode: {self.micro_use}")

        return micro_h

    def get_xrf_embed(self, xrf_x):
        if self.xrf_use == "train":
            xrf_h = self.xrf_block(xrf_x)
            xrf_h = self.merge(xrf_h)
        elif self.xrf_use == "zeros":
            xrf_h = zeros_like_embedding(xrf_x, self.xrf_dim)
        elif self.xrf_use == "skip":
            xrf_h = None
        else:
            raise ValueError(f"Unknown XRF use mode: {self.xrf_use}")

        return xrf_h

    def get_expt_embed(self, expt_x):
        return self.expt_block(expt_x)

    def forward(self, x):
        if self.expt_kwargs["do"]:
            micro_h, xrf_h, expt_h = x
            expt_h = self.get_expt_embed(expt_h)
        else:
            micro_h, xrf_h = x

        h = []
        micro_h = self.get_micro_embed(micro_h)

        if micro_h is not None:
            h.append(micro_h)

        xrf_h = self.get_xrf_embed(xrf_h)

        if xrf_h is not None:
            h.append(xrf_h)

        if self.expt_kwargs["do"]:
            h.append(expt_h)

        h = torch.concat(h, dim=1)
        h = self.pred_block(h)

        return self.final_layer(h)


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

    def get_micro_embed(self, micro_x):
        if self.pretrain:
            micro_h = self.proc(micro_x, return_tensors="pt", do_rescale=True)[
                "pixel_values"
            ]
            with torch.inference_mode():
                micro_h = self.micro_block(micro_h).last_hidden_state[:, 0, :]
        else:
            micro_h = self.micro_block(micro_x)
            micro_h = self.merge(micro_h)
        return micro_h

    def get_xrf_embed(self, xrf_x):
        xrf_h = self.xrf_block(xrf_x)
        xrf_h = self.merge(xrf_h)

        return xrf_h

    def get_expt_embed(self, expt_x):
        return self.expt_block(expt_x)

    def forward(self, x):
        if self.expt_kwargs["do"]:
            micro_h, xrf_h, expt_h = x
            expt_h = self.get_expt_embed(expt_h)
        else:
            micro_h, xrf_h = x

        micro_h = self.get_micro_embed(micro_h)
        xrf_h = self.get_xrf_embed(xrf_h)

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
        self.micro_use = foundation_model_kwargs.get("use", "train")
        self.xrf_use = xrf_kwargs.get("use", "train")

        name = foundation_model_kwargs.get("name", "dinov3-vits16-pretrain-lvd1689m")
        self.name = f"facebook/{name}"

        self.micro_embed = foundation_model_kwargs.get("use", "train")
        if self.micro_use in ["train", "zeros"]:
            self.model_hidden_len = foundation_model_kwargs.get("hidden", 384)
            self.model = AutoModel.from_pretrained(self.name, device_map="auto")
            self.pretrain = True
            self.embedding_type = foundation_model_kwargs.get("embedding_type", "cls")
            self.proc = AutoImageProcessor.from_pretrained(self.name)
            self.frozen_layers = foundation_model_kwargs.get("unfrozen_layers", 0)
            unfreeze_except_lastN(self.model, self.frozen_layers)
        elif self.micro_use in ["skip"]:
            self.model_hidden_len = 0

        # for param in self.micro_block.parameters():
        #     param.requires_grad = False

        self.act = nn.ReLU

        if self.xrf_use in ["train", "zeros"]:
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
            xrf_dim_mult = 1
        elif self.xrf_use in ["skip"]:
            xrf_dim_mult = 0

        if merge == "flatten":
            self.merge = [nn.Identity()]
            self.xrf_dim = +(xrf_kwargs["final_dim"] ** 2) * xrf_kwargs["layers"][-1]
        elif merge == "global":
            self.merge = [nn.AdaptiveAvgPool2d(1)]
            self.xrf_dim = xrf_kwargs["layers"][-1]
        else:
            raise KeyError("Merge technique hasn't been specified")

        self.xrf_dim = self.xrf_dim * xrf_dim_mult
        self.final_dim = self.model_hidden_len + self.xrf_dim
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

    def get_micro_embed(self, micro_x):
        if self.micro_use == "train":
            micro_h = self.proc(micro_x, return_tensors="pt", do_rescale=True)[
                "pixel_values"
            ]

            context = (
                torch.inference_mode() if self.frozen_layers == 0 else nullcontext()
            )

            with context:
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
        elif self.micro_use == "zeros":
            micro_h = zeros_like_embedding(micro_x, self.model_hidden_len)
        elif self.micro_use == "skip":
            micro_h = None
        else:
            raise ValueError(f"Unknown micro use mode: {self.micro_use}")

        return micro_h

    def get_xrf_embed(self, xrf_x):
        if self.xrf_use == "train":
            xrf_h = self.xrf_block(xrf_x)
            xrf_h = self.merge(xrf_h)
        elif self.xrf_use == "zeros":
            xrf_h = zeros_like_embedding(xrf_x, self.xrf_dim)
        elif self.xrf_use == "skip":
            xrf_h = None
        else:
            raise ValueError(f"Unknown XRF use mode: {self.xrf_use}")

        return xrf_h

    def get_expt_embed(self, expt_x):
        return self.expt_block(expt_x)

    def forward(self, x):
        if self.expt_kwargs["do"]:
            micro_h, xrf_h, expt_h = x
            expt_h = self.get_expt_embed(expt_h)
        else:
            micro_h, xrf_h = x

        h = []
        micro_h = self.get_micro_embed(micro_h)

        if micro_h is not None:
            h.append(micro_h)

        xrf_h = self.get_xrf_embed(xrf_h)

        if xrf_h is not None:
            h.append(xrf_h)

        if self.expt_kwargs["do"]:
            h.append(expt_h)

        h = torch.concat(h, dim=1)
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

    def get_micro_embed(self, micro_x):
        micro_h = self.micro_block(micro_x)
        micro_h = self.merge(micro_h)

        return micro_h

    def get_xrf_embed(self, xrf_x):
        xrf_h = self.xrf_block(xrf_x)
        xrf_h = self.merge(xrf_h)

        return xrf_h

    def get_expt_embed(self, expt_x):
        return self.expt_block(expt_x)

    def forward(self, x):
        if self.expt_kwargs["do"]:
            micro_h, xrf_h, expt_h = x
            expt_h = self.get_expt_embed(expt_h)
        else:
            micro_h, xrf_h = x

        micro_h = self.get_micro_embed(micro_h)
        xrf_h = self.get_xrf_embed(xrf_h)

        h = torch.concat([micro_h, xrf_h], 1)
        if self.expt_kwargs["do"]:
            h = torch.concat([h, expt_h], 1)

        h = self.pred_block(h)

        return self.final_layer(h)
