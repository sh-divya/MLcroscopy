from torchvision import models, transforms
from torch.utils.data import DataLoader

from pathlib import Path, PosixPath

from mlcroscopy.data import dataset


def pretrained_model_loader(data_path, source, model, size_handler):
    if isinstance(data_path, PosixPath):
        pass
    else:
        data_path = Path(data_path)

    model, inp_size = get_model(model, size_handler)

    if size_handler == "downscale":
        size_handler = transforms.Resize(
            [
                inp_size[0],
            ]
        )
    elif size_handler == "randomcrop":
        size_handler = transforms.RandomCrop(inp_size[0])
    elif size_handler == "centercrop":
        size_handler = transforms.CenterCrop(inp_size[0])
    elif size_handler == "original":
        pass
    else:
        raise ValueError(
            f"{size_handler} method for handling image / embed size is not implemented"
        )

    loader = get_loader(data_path, source, size_handler)

    return model, loader


def get_model(model, size_handler):
    input_size = None
    if model.lower() == "inception":
        model = models.get_model("inception_v3", weights="DEFAULT")
        if size_handler in ["downscale", "randomcrop", "centercrop"]:
            input_size = [299, 299, 3]
    return model, input_size


def get_loader(data_path, source, size_handler):
    if isinstance(size_handler, str):
        transforms = None
    else:
        transforms = size_handler

    if source == "Microscopy":
        data = dataset.Microscopy_OG(data_path, transforms)
    elif source == "XRF":
        data = dataset.XRF_OG(data_path, transforms)

    loader = DataLoader(data, len(data), shuffle=False)

    return loader
