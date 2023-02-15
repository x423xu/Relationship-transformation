import torch

from typing import Dict, Union
import pytorch_lightning as pl
import torch.nn as nn

from metrics import make_losses, make_metrics
from .RelTrans import RelTrans


"""
pytorch lightning module
"""


class PLPredictionModule(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = RelTrans(args)
        self.losses, self.loss_weights = make_losses(args)
        # self.losses = nn.MSELoss()

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
        R_tilde, pts3d = self.model(batch)
        mask = self._get_correpondence(pts3d, batch)
        loss = self.comput_loss(R_tilde, mask, batch)
        self.log("train_loss", loss, on_step=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):

        R_tilde, pts3d = self.model(batch)
        mask = self._get_correpondence(pts3d, batch)
        loss = self.comput_loss(R_tilde, mask, batch)
        val_metrics = make_metrics(self.args)
        values = self.compute_metrics(
            val_metrics, R_tilde, batch["rel_features"][1], mask
        )
        self.log(self.args.monitor, loss, on_step=False, on_epoch=True, rank_zero_only=True)
        self.log("val_metrics", values, on_step=True, on_epoch=True, rank_zero_only=True)
        return loss

    def test_step(self, batch, batch_idx):
        return 0

    """
    compute loss and add them together
    """

    def comput_loss(self, output, mask, batch):
        loss = 0
        for loss_fn, w in zip(self.losses, self.loss_weights):
            loss += w * loss_fn(output, mask, batch)
        return loss

    def compute_metrics(self, metrics, prediction, groundtruth, mask):
        values = {}
        for k, m in metrics.items():
            value = m.compute(prediction, groundtruth, mask)
            values.update({k: value})
        return values

    def _get_correpondence(self, pts3D_orig, batch):
        def _bbox_map(bbox, pts3D, W):
            pts3D = pts3D.clone().detach()
            bbox = bbox.clone().detach()
            pts3D[:, :, 1] = (-pts3D[:, :, 1]) * W // 2 + W // 2
            pts3D[:, :, 0] = (-pts3D[:, :, 0]) * W // 2 + W // 2
            pts3D[pts3D < 0] = 0
            pts3D[pts3D > (W - 1)] = W - 1
            pts3D = pts3D.to(torch.int64)

            bbox = bbox * W
            bbox = bbox.to(torch.int64)
            # b, n, c = bbox.shape
            # bbox = bbox.view(-1, c)
            sub_x1y1 = bbox[:, :, 1] * W + bbox[:, :, 0]
            sub_x2y2 = bbox[:, :, 3] * W + bbox[:, :, 2]
            obj_x1y1 = bbox[:, :, 5] * W + bbox[:, :, 4]
            obj_x2y2 = bbox[:, :, 7] * W + bbox[:, :, 6]

            new_bbox = [
                torch.gather(pts3D[:, :, 0], 1, sub_x1y1).unsqueeze(-1),
                torch.gather(pts3D[:, :, 1], 1, sub_x1y1).unsqueeze(-1),
                torch.gather(pts3D[:, :, 0], 1, sub_x2y2).unsqueeze(-1),
                torch.gather(pts3D[:, :, 1], 1, sub_x2y2).unsqueeze(-1),
                torch.gather(pts3D[:, :, 0], 1, obj_x1y1).unsqueeze(-1),
                torch.gather(pts3D[:, :, 1], 1, obj_x1y1).unsqueeze(-1),
                torch.gather(pts3D[:, :, 0], 1, obj_x2y2).unsqueeze(-1),
                torch.gather(pts3D[:, :, 1], 1, obj_x2y2).unsqueeze(-1),
            ]
            new_bbox = torch.cat(new_bbox, dim=-1)
            return new_bbox

        def _match(batch):
            idx_pairs1 = batch["idx_pairs"][0]
            idx_pairs2 = batch["idx_pairs"][1]
            idxp1 = idx_pairs1.unsqueeze(2)
            idxp2 = idx_pairs2.unsqueeze(1)
            mask = (idxp1 == idxp2).all(-1)
            mask = mask.nonzero()
            return mask

        # bbox1 = batch["bbox"][0]
        # bbox1_mapped = _bbox_map(bbox1, pts3D_orig, self.args.W)
        # bbox2 = batch["bbox"][1]
        mask = _match(batch)
        return mask
