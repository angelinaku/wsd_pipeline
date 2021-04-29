from typing import Tuple, List

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer


class BasicDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        """
        Class to create pytorch dataset from initial data
        :param data: pandas dataframe with text and its label
        """

        self.data = data
        self.data_length = len(data)

        self.tokens = self.data.question.to_list()
        self.labels = np.array(self.data.need_clari.to_list())

    def __getitem__(self, idx: int) -> Tuple[List, int]:

        return self.tokens[idx], self.labels[idx]

    def __len__(self) -> int:

        return self.data_length


class BasicCollator:
    def __init__(self, model_name: str = 'bert-base-uncased'):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, batch):

        text, labs = zip(*batch)

        output_dict = self.tokenizer.batch_encode_plus(
            list(text),
            add_special_tokens=True,
            pad_to_max_length=True,
            padding='longest',
            return_tensors='pt',
            return_attention_mask=True,
        )

        labs = np.array(list(labs))
        labs = torch.as_tensor(labs)

        token_type_ids = None

        return output_dict['input_ids'], output_dict['attention_mask'], token_type_ids, labs


class SequencePairsDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        """
        Class to create pytorch dataset from initial data for sequence pair classification task
        :param data: pandas dataframe with text and its label
        """

        self.data = data
        self.data_length = len(data)

        self.queries = self.data['query'].to_list()
        self.responses = self.data['response'].to_list()
        self.sequence_pairs = []
        for i, j in zip(self.queries, self.responses):
            self.sequence_pairs.append((i, j))

        self.labels = np.array(self.data.need_clari.to_list())

    def __getitem__(self, idx: int) -> Tuple[Tuple[str, str], int]:

        return self.sequence_pairs[idx], self.labels[idx]

    def __len__(self) -> int:

        return self.data_length


class SequencePairsCollator:
    def __init__(self, model_name: str = 'DeepPavlov/rubert-base-cased-conversational'):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, batch):

        text, labs = zip(*batch)

        output_dict = self.tokenizer.batch_encode_plus(
            list(text),
            add_special_tokens=True,
            pad_to_max_length=True,
            padding='longest',
            return_tensors='pt',
            return_attention_mask=True,
        )

        labs = np.array(list(labs))
        labs = torch.as_tensor(labs)

        return output_dict['input_ids'], output_dict['attention_mask'], output_dict['token_type_ids'], labs