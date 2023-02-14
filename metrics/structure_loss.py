from .base import Metrics


class StructureLoss(Metrics):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def compute(self, output, pts3d, batch):
        return 0

    def __call__(self, output, pts3d, batch):
        return self.compute(output, pts3d, batch)
