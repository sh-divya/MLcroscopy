from pathlib import PosixPath, Path
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
import pytorch_lightning as pl


class ScopyLightning(pl.LightningDataModule):
    def __init__(self, trainset, valset, testset, batch_size, num_workers):
        super().__init__()

        self.data_train = trainset
        self.data_val = valset
        self.data_test = testset

        self.dl_args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": False,
        }

    def setup(self, stage):
        return super().setup(stage)

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, **self.dl_args)

    def val_dataloader(self):
        if self.data_val:
            return DataLoader(self.data_val, shuffle=False, **self.dl_args)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, **self.dl_args)


class ScopyTensor(Dataset):
    elements = ["ag_la", "cu", "cl", "s"]
    targets = ["FE (Eth)", "FE (CO)"]

    def __init__(
        self,
        root,
        split_fptr,
        micro_transform=None,
        xrf_transform=None,
        ytransform=None,
    ):
        self.root = Path(root)
        self.csv = pd.read_csv(self.root / "triplet.csv")
        self.microscopy = self.root / "Microscopy"
        self.xrf = pickle.load(open(self.root / "XRF" / "xrf_raw.pkl", "rb"))
        self.idx = [int(i) for i in open(self.root / split_fptr, "r")]
        self.micro_transform = micro_transform if micro_transform else False
        self.xrf_transform = xrf_transform if xrf_transform else False
        self.ytransform = ytransform
        if self.ytransform:
            self.ytransform = self.get_triplet_stats()

    def stats(self):
        targets = self.csv[self.targets]
        return targets.min(), targets.max()

    def get_triplet_stats(self):
        grouped = self.csv.groupby("triplet")
        mean = grouped.mean()[self.targets]
        std = grouped.std()[self.targets]
        return {"mean": mean, "std": std}

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        idx = self.idx[idx]
        triplet_id = self.csv.loc[idx, "triplet"]
        micro_fptr = self.microscopy / f"{idx}.png"
        micro_img = decode_image(micro_fptr)[:, :-150, :].to(float)
        if self.micro_transform:
            micro_img = self.micro_transform(micro_img)

        target = torch.tensor(self.csv.loc[idx][self.targets].values)
        if self.ytransform:
            mean = self.ytransform["mean"].loc[triplet_id].values
            std = self.ytransform["std"].loc[triplet_id].values
            target = torch.normal(torch.tensor(mean), torch.tensor(std))
            # target = torch.max(target, 0 dim=1)
        else:
            pass

        xrf_data = self.xrf[idx]
        tmp = []
        for k, v in xrf_data.items():
            tmp.append(torch.tensor(v).unsqueeze(0))
        xrf_data = torch.concat(tmp, 0)
        if self.xrf_transform:
            xrf_data = self.xrf_transform(xrf_data)
        return (micro_img.to(torch.float), xrf_data.to(torch.float)), target.to(
            torch.float
        )


class Microscopy_OG(Dataset):
    def __init__(self, root, transform):
        if isinstance(root, PosixPath):
            pass
        elif isinstance(root, str):
            root = Path(root)
        else:
            raise ValueError(f"Root variable of type {type(root)} not supported")
        self.root = root / "Microscopy"
        self.csv = root / "Characterization_data.xlsx"
        self.transform = transform
        self.experimental_data = pd.read_excel(self.csv, index_col=0, skiprows=[1])[
            "FE (CO)"
        ].values

    def __len__(self):
        return len(self.experimental_data)

    def __getitem__(self, idx):
        img_path = self.root / f"{idx}.png"
        expmnt_val = self.experimental_data[idx]
        img = decode_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        # print(type(img), type(expmnt_val))
        # raise Exception
        return img, expmnt_val


class XRF_OG(Dataset):
    def __init__(self, root, transform):
        if isinstance(root, PosixPath):
            pass
        elif isinstance(root, str):
            root = Path(root)
        else:
            raise ValueError(f"Root variable of type {type(root)} not supported")
        self.root = root / "XRF"
        self.csv = root / "Characterization_data.xlsx"
        self.transform = transform
        self.img_names = {
            "Ag": "$IDX$_ag_la-map.png",
            "Cl": "$IDX$_cl-map.png",
            "Cu": "$IDX$_cu-map.png",
            "S": "$IDX$_s-map.png",
        }
        self.experimental_data = pd.read_excel(self.csv, index_col=0, skiprows=[1])[
            "FE (CO)"
        ].values

    def __len__(self):
        return len(self.experimental_data)

    def __getitem__(self, idx):
        expmnt_val = self.experimental_data[idx]
        img = {}
        for k, v in self.img_names.items():
            sample = decode_image(self.root / v.replace("$IDX$", str(idx)))
            if self.transform is not None:
                img[k] = self.transform(sample)
        return img, expmnt_val


class OnlyExpt(Dataset):
    targets = ["FE (Eth)", "FE (CO)"]

    def __init__(self, root, split_fptr, xstats, ytransform=False):
        self.root = Path(root)
        self.csv = pd.read_csv(self.root / "triplet.csv")
        self.idx = [int(i) for i in open(self.root / split_fptr, "r")]
        self.xstats = xstats
        if ytransform:
            self.ytransform = self.get_triplet_stats()
        else:
            self.ytransform = ytransform

    def __len__(self):
        return len(self.idx)

    def stats(self):
        targets = self.csv[self.targets]
        return targets.min(), targets.max()

    def get_triplet_stats(self):
        grouped = self.csv.groupby("triplet")
        mean = grouped.mean()[self.targets]
        std = grouped.std()[self.targets]
        return {"mean": mean, "std": std}

    def __getitem__(self, idx):
        idx = self.idx[idx]
        x = torch.tensor(self.csv.iloc[idx, :-3].values)
        y = torch.tensor(self.csv.iloc[idx, -3:-1].values)
        if self.xstats:
            x = (x - self.xstats["mean"]) / self.xstats["std"]

        triplet_id = self.csv.iloc[idx, -1]
        if self.ytransform:
            mean = self.ytransform["mean"].loc[triplet_id].values
            std = self.ytransform["std"].loc[triplet_id].values
            y = torch.normal(torch.tensor(mean), torch.tensor(std))
        else:
            pass
        return x.to(torch.float32), y.to(torch.float32)


class ImgPlusExpt(ScopyTensor):

    def __init__(
        self,
        root,
        split_fptr,
        expt_transform=None,
        micro_transform=None,
        xrf_transform=None,
        ytransform=None,
    ):
        super().__init__(root, split_fptr, micro_transform, xrf_transform, ytransform)
        self.xstats = expt_transform

    def __getitem__(self, idx):
        img, y = super().__getitem__(idx)
        img_micro, img_xrf = img
        idx = self.idx[idx]
        expt_x = torch.tensor(self.csv.iloc[idx, :-3].values).to(torch.float32)

        return (img_micro, img_xrf, expt_x), y
