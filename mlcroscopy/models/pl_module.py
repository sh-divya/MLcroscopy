import torch
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection


class MultiRegress(pl.LightningModule):
    def __init__(self, model, loss, lr, device):
        super().__init__()
        self.model = model  # .to(device)
        self.loss = loss
        self.lr = lr
        self.train_metric = MetricCollection(
            {"MAE": MeanAbsoluteError(), "MSE": MeanSquaredError()}, prefix="train_"
        )
        self.valid_metric = MetricCollection(
            {"MAE": MeanAbsoluteError(), "MSE": MeanSquaredError()}, prefix="valid_"
        )
        self.best_mae = 1e5
        self.best_mse = 1e5
        # self.dev = device

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x1 = x1.to(self.dev)
        # x2 = x2.to(self.dev)
        # y = y.to(self.dev)
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
        self.train_metric.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.loss(out, y)
        val_metric = self.valid_metric(out, y)

        self.log("val_loss", loss)
        self.log("Val MAE", val_metric["valid_MAE"])
        self.log("Val MSE", val_metric["valid_MSE"])

        return loss

    def on_validation_epoch_end(self):
        total_val_metric = self.valid_metric.compute()

        self.log("total_val_mae", total_val_metric["valid_MAE"])
        self.log("total_val_mse", total_val_metric["valid_MSE"])

        if total_val_metric["valid_MAE"] < self.best_mae:
            self.best_mae = total_val_metric["valid_MAE"]
        if total_val_metric["valid_MSE"] < self.best_mse:
            self.best_mse = total_val_metric["valid_MSE"]
        self.valid_metric.reset()

    def on_validation_end(self):
        try:
            self.logger.experiment.summary["Best MAE"] = self.best_mae
            self.logger.experiment.summary["Best MSE"] = self.best_mse
        except TypeError:
            print("Best MAE", self.best_mae)
            print("Best MSE", self.best_mse)
            pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=4
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "total_val_mae"},
        }
