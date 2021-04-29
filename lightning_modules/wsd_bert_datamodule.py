from typing import Dict, Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
import pandas as pd
from torch.utils.data import DataLoader

from utils.utils import load_obj


class DatamoduleBasic(pl.LightningDataModule):
    def __init__(self, hparams: Dict, conf: DictConfig):
        super().__init__()

        self.conf = conf
        self.hparams = hparams

        self.batch_size = self.conf.training.batch_size
        self.num_workers = self.conf.training.num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # Load train-test-validation datasets
        self.train_df = pd.read_csv(f'{self.conf.data.main_folder}{self.conf.data.train_data_name}', sep='|')
        self.test_df = pd.read_csv(f'{self.conf.data.main_folder}{self.conf.data.test_data_name}', sep='|')
        self.valid_df = pd.read_csv(f'{self.conf.data.main_folder}{self.conf.data.valid_data_name}', sep='|')

        self.collator = load_obj(self.conf.training.collator.name)(model_name=self.conf.model.model_name)

        self.torch_dataset = load_obj(self.conf.training.torch_dataset_class.name)
        self.train_dataset = self.torch_dataset(self.train_df)
        self.test_dataset = self.torch_dataset(self.test_df)
        self.valid_dataset = self.torch_dataset(self.valid_df)

    def train_dataloader(self) -> DataLoader:

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader

    def val_dataloader(self) -> DataLoader:

        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self) -> DataLoader:

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader
