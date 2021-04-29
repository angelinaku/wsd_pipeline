import json
from typing import Dict, Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from datasets.dataset_collator import WSDDataset, WSDCollatorELMo
from utils.text_processing_utils import ELMo_Embedder


class ELMoDatamoduleWSD(pl.LightningDataModule):

    def __init__(self, hparams: Dict[str, float], conf: DictConfig):
        super().__init__()

        self.conf = conf
        self.hparams = hparams

        self.bs = self.conf.training.batch_size
        self.num_works = self.conf.training.num_workers

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):

        # Loading Train-Test-Validation Data
        with open(f'{self.conf.data.main_folder}{self.conf.data.train_data_name}', 'r', encoding='utf-8') as f:
            train_wsd_data = json.load(f)

        with open(f'{self.conf.data.main_folder}{self.conf.data.valid_data_name}', 'r', encoding='utf-8') as f:
            valid_wsd_data = json.load(f)

        with open(f'{self.conf.data.main_folder}{self.conf.data.test_data_name}', 'r', encoding='utf-8') as f:
            test_wsd_data = json.load(f)

        # Loading file with mapping of tags to their ids
        with open(f'{self.conf.data.main_folder}{self.conf.data.tag_to_idx_name}', 'r', encoding='utf-8') as f:
            self.tag_to_idx = json.load(f)

        self.word_to_idx = {}

        self.wsd_collator = WSDCollatorELMo(percentile=self.conf.training.collator.percent,
                                            pad_value=self.tag_to_idx['PAD'],
                                            max_possible_len=self.conf.training.collator.max_seq_length,
                                            pad_type=self.conf.training.collator.pad_type)

        self.train_dataset = WSDDataset(train_wsd_data, self.conf, self.word_to_idx, self.tag_to_idx)
        self.validation_dataset = WSDDataset(valid_wsd_data, self.conf, self.word_to_idx, self.tag_to_idx)
        self.test_dataset = WSDDataset(test_wsd_data, self.conf, self.word_to_idx, self.tag_to_idx)

        if self.conf.data.vectorization is True:
            self.embedder = ELMo_Embedder(embeddings_path=self.conf.data.embeddings_path)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.bs,
            num_workers=self.num_works,
            collate_fn=self.wsd_collator,
            shuffle=False
        )
        return train_loader

    def val_dataloader(self, *args, **kwargs) -> DataLoader:

        valid_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.bs,
            num_workers=self.num_works,
            collate_fn=self.wsd_collator,
            shuffle=False
        )
        return valid_loader

    def test_dataloader(self, *args, **kwargs) -> DataLoader:

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.bs,
            num_workers=self.num_works,
            collate_fn=self.wsd_collator,
            shuffle=False
        )
        return test_loader
