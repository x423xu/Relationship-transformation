import torch

from typing import Dict, Union
import pytorch_lightning as pl
import torch.nn as nn

from metrics import make_metrics
from .networks.z_buffermodel import ZbufferModelPts


class RelTrans(nn.Module):


    def __init__(
        self,args
    ):

        super().__init__()
        self.args = args
        self.model = ZbufferModelPts(args)

    def forward(self, R: torch.Tensor, B: torch.Tensor) -> Union[torch.Tensor, Dict]:

        Ro, Bo,  = self.model(R, B)

        return Ro,Bo


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
        [R1, R2] = batch['R']
        [B1, B2] = batch['B']
        R_tilde, B_tilde = self.model(R1, B1)
        rel_loss = self.losses(R_tilde, R2)
        box_loss = self.losses(B_tilde, B2)
        loss = rel_loss+box_loss
        self.log("train_loss", loss, on_step=True)
        return loss


    def validation_step(self, batch, batch_idx):

        [target_imgs,ref_imgs] = batch['images']
        K = batch['K']
        [P1, P2] = batch['P']
        [Pinv1, Pinv2] = batch['Pinv']
        [R1, R2] = batch['R']
        [B1, B2] = batch['B']
        R_tilde, B_tilde = self.model(R1, B1)
        rel_loss = self.losses(R_tilde, R2)
        box_loss = self.losses(B_tilde, B2)
        loss = rel_loss+box_loss
        self.log(self.args.monitor, loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        return 0
    """
    compute loss and add them together
    """

