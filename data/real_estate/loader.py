import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .utils import get_dataset


class RealEstateRelationshipDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = get_dataset("train", self.args)
            self.dataset_val = get_dataset("val", self.args)
        if stage == "test" or stage is None:
            self.dataset_test = get_dataset("test", self.args)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            num_workers=self.args.num_workers,
            batch_size=self.args.train_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            prefetch_factor = self.args.pre_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            num_workers=self.args.num_workers,
            batch_size=self.args.val_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            prefetch_factor = self.args.pre_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            num_workers=self.args.num_workers,
            batch_size=self.args.test_size,
            shuffle=False,
            pin_memory=False,
        )

    def predict_dataloader(self):
        pass
