from typing import List, Dict, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from utils.text_processing_utils import pad_sequences


class WSDDataset(Dataset):

    def __init__(self, wsd_data: List, conf: DictConfig, word_to_idx: Dict, tag_to_idx: Dict):
        """
        Args:
            wsd_data: data
            word_to_idx: mapping of words do indexes
            conf: config with parameters
            tag_to_idx: mapping of tags do indexes
        """
        self.wsd_data = wsd_data
        self.data_length = len(wsd_data)
        self.conf = conf
        if word_to_idx != {}:
            self.tokens = np.array(
                [[word_to_idx[w] if w in word_to_idx.keys() else 1 for w in line[0]] for line in wsd_data]
            )
        else:
            self.tokens = [[w for w in line[0]] for line in wsd_data]

        self.labels = np.array([[tag_to_idx[w] for w in line[1]] for line in wsd_data])

    def __getitem__(self, idx: int) -> Tuple[np.array, int, np.array]:

        return self.tokens[idx], len(self.tokens[idx]), self.labels[idx]

    def __len__(self) -> int:
        return self.data_length


class WSDCollator:
    def __init__(self, percentile: int = 100, pad_value: int = 0, max_possible_len: int = 200, pad_type: str = 'post'):

        self.percentile = percentile
        self.pad_value = pad_value
        self.max_possible_len = max_possible_len
        self.pad_type = pad_type

    def __call__(self, batch):
        tokens, lens, labels = zip(*batch)
        lens = np.array(lens)

        max_len = min(int(np.percentile(lens, self.percentile)), self.max_possible_len)

        tokens = torch.tensor(
            pad_sequences(tokens, maxlen=max_len, padding=self.pad_type, value=self.pad_value), dtype=torch.long
        )

        lengths = torch.tensor([min(i, max_len) for i in lens], dtype=torch.long)

        labels = torch.tensor(
            pad_sequences(labels, maxlen=max_len, padding=self.pad_type, value=self.pad_value), dtype=torch.long
        )

        return tokens, lengths, labels


class WSDCollatorELMo:
    def __init__(self, percentile: int = 100, pad_value: int = 0, max_possible_len: int = 200, pad_type: str = 'post'):

        self.percentile = percentile
        self.pad_value = pad_value
        self.max_possible_len = max_possible_len
        self.pad_type = pad_type

    def __call__(self, batch):
        tokens, lens, labels = zip(*batch)
        lens = np.array(lens)

        max_len = max(lens)

        # tokens = torch.tensor(
        #     pad_sequences(tokens, maxlen=max_len, padding=self.pad_type, value=self.pad_value), dtype=torch.long
        # )

        lengths = torch.tensor([min(i, max_len) for i in lens], dtype=torch.long)

        labels = torch.tensor(
            pad_sequences(labels, maxlen=max_len, padding=self.pad_type, value=self.pad_value), dtype=torch.long
        )

        return tokens, lengths, labels
