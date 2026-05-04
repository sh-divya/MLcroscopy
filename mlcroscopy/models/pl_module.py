import torch
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from mlcroscopy.models.archs import ScopyDino, ScopyRes


class MultiRegress(pl.LightningModule):

    dino_attr_name = {"ScopyDino": "model", "ScopyRes": "micro_block"}

    def __init__(self, model, loss, lr, device, hparams=None):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.has_val = hparams["model"]["has_val"]
        self.train_metric = MetricCollection(
            {"MAE": MeanAbsoluteError(), "MSE": MeanSquaredError()}, prefix="train_"
        )
        self.valid_eth = MetricCollection(
            {"MAE": MeanAbsoluteError(), "MSE": MeanSquaredError()}, prefix="valid_eth_"
        )

        self.valid_co = MetricCollection(
            {"MAE": MeanAbsoluteError(), "MSE": MeanSquaredError()}, prefix="valid_co_"
        )

        
        self.best_eth_mae = 1e5
        self.best_eth_mse = 1e5
        self.best_co_mae = 1e5
        self.best_co_mse = 1e5
        self.save_hyperparameters("hparams", ignore=["model"])

    def training_step(self, batch, batch_idx):
        x, y = batch

        out = self.model(x)
        loss = self.loss(out, y)
        train_batch = self.train_metric(out, y)

        lr = self.optimizers().param_groups[0]["lr"]

        self.log("train_loss", loss)
        self.log("Train MAE", train_batch["train_MAE"])
        self.log("Train MSE", train_batch["train_MSE"])
        self.log("Learning rate", lr)

        return loss

    def on_train_epoch_end(self):
        train_epoch = self.train_metric.compute()
        self.log("total_train_MAE", train_epoch["train_MAE"])
        self.log("total_train_MSE", train_epoch["train_MSE"])
        self.train_metric.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.loss(out, y)

        self.log("val_loss", loss)


        val_metric = self.valid_eth(out[:, 0], y[:, 0])

        self.log("Val Eth MAE", val_metric["valid_eth_MAE"])
        self.log("Val Eth MSE", val_metric["valid_eth_MSE"])

        val_metric = self.valid_co(out[:, 1], y[:, 1])

        self.log("Val CO MAE", val_metric["valid_co_MAE"])
        self.log("Val CO MSE", val_metric["valid_co_MSE"])

        return loss

    def on_validation_epoch_end(self):
        total_val_eth = self.valid_eth.compute()
        total_val_co = self.valid_co.compute()

        self.log("total_val_eth_mae", total_val_eth["valid_eth_MAE"])
        self.log("total_val_eth_mse", total_val_eth["valid_eth_MSE"])
        self.log("total_val_co_mae", total_val_co["valid_co_MAE"])
        self.log("total_val_co_mse", total_val_co["valid_co_MSE"])

        if total_val_eth["valid_eth_MAE"] < self.best_eth_mae:
            self.best_eth_mae = total_val_eth["valid_eth_MAE"]
        if total_val_eth["valid_eth_MSE"] < self.best_eth_mse:
            self.best_eth_mse = total_val_eth["valid_eth_MSE"]
        if total_val_co["valid_co_MAE"] < self.best_co_mae:
            self.best_co_mae = total_val_co["valid_co_MAE"]
        if total_val_co["valid_co_MSE"] < self.best_co_mse:
            self.best_co_mse = total_val_co["valid_co_MSE"]

        self.valid_eth.reset()
        self.valid_co.reset()

    def on_validation_end(self):
        try:
            self.logger.experiment.summary["Best Eth MAE"] = self.best_eth_mae
            self.logger.experiment.summary["Best Eth MSE"] = self.best_eth_mse
            self.logger.experiment.summary["Best CO MAE"] = self.best_co_mae
            self.logger.experiment.summary["Best CO MSE"] = self.best_co_mse
        except TypeError:
            pass
            # print("Best Eth MAE", self.best_eth_mae)
            # print("Best Eth MSE", self.best_eth_mse)
            # print("Best CO MAE", self.best_co_mae)
            # print("Best CO MSE", self.best_co_mse)

    def configure_optimizers(self):
        model_type = type(self.model).__name__
        if getattr(self.model, "pretrain", False):
            unfrozen_params = []
            attr_name = self.dino_attr_name[model_type]
            for p in getattr(self.model, attr_name).parameters():
                if p.requires_grad:
                    print("Only if layers are unfrozen")
                    unfrozen_params.append(p)
            para_list = [{"params": unfrozen_params, "lr": self.lr / 10}]
            other_params = []
            for name, p in self.model.named_parameters():
                if not name.startswith(f"{attr_name}."):
                    if p.requires_grad:
                        other_params.append(p)
            para_list.append({"params": other_params, "lr": self.lr})

        else:
            para_list = [{"params": [p for p in self.parameters() if p.requires_grad], "lr": self.lr}]
        
        optimizer = optim.Adam(para_list)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=4
        )

        if self.has_val:    
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "total_val_eth_mae",
                },
            }
        else:
            return optimizer
