from .base import Metrics
import torch
import numpy as np
REL_DIM=51
class RecallK(Metrics):
    def __init__(self, K, args) -> None:
        super().__init__()
        self.K = K
        self.args = args

    def compute(self, prediction, ground_truth, mask):
        y = ground_truth[mask[:,0], mask[:,2], :]
        y_tilde = prediction[mask[:,0], mask[:,1], :]
        recall = []
        for b in range(self.args.val_size):
            y_b = y[mask[:,0]==b]
            y_t_b = y_tilde[mask[:,0]==b]
            y_index = torch.argsort(torch.max(y_b, -1)[0], descending=True)
            y__tilde_index = torch.argsort(y_t_b.max(-1)[0], descending=True)
            y_class = y_b.argmax(-1)[y_index[:self.K]].detach().cpu().numpy()
            y_tilde_class = y_t_b.argmax(-1)[y__tilde_index[:self.K]].detach().cpu().numpy()
            intsect = np.intersect1d(y_class, y_tilde_class)
            recall.append(len(intsect)/self.K)
        recall = np.array(recall).mean()
        return recall
    
    def __call__(self,prediction, ground_truth, mask):
        return self.compute(prediction, ground_truth, mask)