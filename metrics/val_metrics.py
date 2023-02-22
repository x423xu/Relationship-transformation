from .base import Metrics
import torch
import numpy as np

REL_DIM = 51


class RecallK(Metrics):
    def __init__(self, K, args) -> None:
        super().__init__()
        self.K = K
        self.args = args

    def compute(self, prediction, ground_truth, mask):
        batch_size = prediction.shape[0]
        y = ground_truth[mask[:, 0], mask[:, 2], :]
        y_tilde = prediction[mask[:, 0], mask[:, 1], :]
        recall = []
        for b in range(batch_size):
            y_b = y[mask[:, 0] == b]
            y_t_b = y_tilde[mask[:, 0] == b]
            
            y_index = torch.argsort(torch.max(y_b, -1)[0], descending=True)
            y_tilde_index = torch.argsort(y_t_b.max(-1)[0], descending=True)
            # y_t_positive = (torch.max(y[y_tilde_index[:self.K],:], -1)[0]>0.5).sum().cpu().numpy()
            # if y_t_positive<=0:
            #     continue
            y_class = y_b.argmax(-1)[y_index[: self.K]]
            y_tilde_class = y_t_b.argmax(-1)[y_tilde_index[: self.K]]
            intsect = y_class.unsqueeze(1) == y_tilde_class.unsqueeze(0) 
            intsect = intsect.any(0).sum().detach().cpu().numpy()
            recall.append(intsect / self.K)
        recall = np.array(recall).mean()
        return recall

    def __call__(self, prediction, ground_truth, mask):
        return self.compute(prediction, ground_truth, mask)


class Precision(Metrics):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, prediction, ground_truth, mask):
        y = ground_truth[mask[:, 0], mask[:, 2], :]
        y_tilde = prediction[mask[:, 0], mask[:, 1], :]
        y_class = y.argmax(-1)
        y_tilde_class = y_tilde.argmax(-1)

        precision = (y_class == y_tilde_class).sum() / (y_class.shape[0])

        return precision

    def __call__(self, prediction, ground_truth, mask):
        return self.compute(prediction, ground_truth, mask)
