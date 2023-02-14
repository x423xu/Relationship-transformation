from .base import Metrics
import torch


class ValueLoss(Metrics):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.distance = torch.nn.MSELoss()

    def compute(self, output, mask, batch):
        ground_truth = batch['rel_features'][1]
        y_tilde = output[mask[:,0], mask[:,1], :]
        y = ground_truth[mask[:,0], mask[:,2], :]
        dist = self.distance(y,y_tilde).mean()
        return dist

    def __call__(self, output, mask, batch):
        return self.compute(output, mask, batch)
