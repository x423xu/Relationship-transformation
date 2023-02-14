import torch

from typing import Dict, Union
import pytorch_lightning as pl
import torch.nn as nn

from metrics import make_metrics
from .RelTrans import RelTrans


"""
pytorch lightning module
"""


class PLPredictionModule(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model = RelTrans(args)
        # self.losses = make_metrics()
        self.losses = nn.MSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.args.scheduler_step,
            gamma=self.args.scheduler_gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "metric_to_track",
                "frequency": self.args.scheduler_frequency,
            },
        }

    def training_step(self, batch, batch_idx):
        [target_imgs,ref_imgs] = batch['images']
        K = batch['K']
        [P1, P2] = batch['P']
        [Pinv1, Pinv2] = batch['Pinv']
        [R1, R2] = batch['rel_features']
        [B1, B2] = batch['bbox']
        R_tilde, pts3d = self.model(batch)
        rel_loss = self.losses(R_tilde, R2)
        # box_loss = self.losses(B_tilde, B2)
        loss = rel_loss
        self.log("train_loss", loss, on_step=True)
        return loss


    def validation_step(self, batch, batch_idx):
        [target_imgs,ref_imgs] = batch['images']
        K = batch['K']
        [P1, P2] = batch['P']
        [Pinv1, Pinv2] = batch['Pinv']
        [R1, R2] = batch['rel_features']
        [B1, B2] = batch['bbox']
        R_tilde, pts3d = self.model(batch)
        rel_loss = self.losses(R_tilde, R2)
        # box_loss = self.losses(B_tilde, B2)
        loss = rel_loss
        self.log(self.args.monitor, loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        return 0
    """
    compute loss and add them together
    """

