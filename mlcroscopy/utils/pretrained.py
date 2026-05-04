import torch
from torch import nn
from hydra.utils import instantiate
import copy


class ScopyPredictor(nn.Module):

    skip_pre_data = ["crop", "color", "flip"]
    src_modals = ["expt", "micro", "xrf"]

    def __init__(self, path_to_ckpt, device, inference=True):
        super().__init__()

        self.inference = inference
        ckpt = torch.load(path_to_ckpt)

        self.device = torch.device(device)
        self.arch_params = ckpt["hyper_parameters"]["hparams"]["model"]
        self.task_params = ckpt["hyper_parameters"]["hparams"]["task"]

        kwargs = {}
        for p in ["batch_norm", "droprate"]:
            if p in self.arch_params["optim"]:
                kwargs[p] = self.arch_params["optim"][p]

        self.model = instantiate(self.arch_params["arch"], **kwargs).to(
            device=self.device, dtype=torch.float16
        )

        weights = self.edit_dict_keys(ckpt["state_dict"])
        self.model.load_state_dict(weights)
        if self.inference:
            self.model.eval()

    def edit_dict_keys(self, state_dict):
        return {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}

    def get_data_proc(self):
        if self.inference:
            transforms = copy.deepcopy(self.task_params["transforms"])
            for src in self.src_modals:
                if src in self.task_params["transforms"]:
                    for trans in self.task_params["transforms"][src]:
                        if trans in self.skip_pre_data:
                            transforms[src].pop(trans)
            return transforms
        else:
            return self.task_params

    def __call__(self, x):
        if self.inference:
            with torch.no_grad():
                return self.model(x)
        else:
            return self.model(x)
