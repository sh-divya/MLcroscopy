import torch
from torch import nn
from hydra.utils import instantiate

class ScopyPredictor(nn.Module):
    def __init__(self, path_to_ckpt, device):
        super().__init__()
        
        ckpt = torch.load(path_to_ckpt)
        
        self.device = None
        self.arch_params = ckpt["hyper_parameters"]["hparams"]["model"]
        self.task_params = ckpt["hyper_parameters"]["hparams"]["task"]
        weights = ckpt["state_dict"]

        kwargs = {}
        for p in ["batch_norm", "droprate"]:
            if p in self.arch_params["optim"]:
                kwargs[p] = self.arch_params["optim"][p]

        model = instantiate(self.arch_params["model"]["arch"], **kwargs).to(torch.float)

        kwargs = {"micro": None, "xrf": None, "expt": None}
        for p in ["norm", "crop", "oneD"]:
            pass


    def __call__(self, x):
        pass