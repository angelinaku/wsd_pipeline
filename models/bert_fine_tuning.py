import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from models.layers import MaskedAverageLayer


class BertForSequenceClassification(nn.Module):
    """
    Simplified version of the same class by HuggingFace.
    See transformers/modeling_distilbert.py in the transformers repository.
    """

    def __init__(self, pretrained_model_name: str,
                 num_classes: int = None, dropout: float = 0.3):
        """
        Args:
            pretrained_model_name (str): HuggingFace model name.
                See transformers/modeling_auto.py
            num_classes (int): the number of class labels
                in the classification task
        """
        super().__init__()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_classes)

        self.bert_model = AutoModel.from_pretrained(pretrained_model_name,
                                                    config=config)
        self.masked_average = MaskedAverageLayer()
        output_size = 1 if num_classes == 2 else num_classes-1
        self.classifier = nn.Linear(config.hidden_size*4, output_size)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, attention_mask=None, target_mask=None):
        """Compute class probabilities for the input sequence.

        Args:
            features (torch.Tensor): ids of each token,
                size ([bs, seq_length]
            attention_mask (torch.Tensor): binary tensor, used to select
                tokens which are used to compute attention scores
                in the self-attention heads, size [bs, seq_length]
            target_mask (torch.Tensor):
        Returns:
            PyTorch Tensor with predicted class probabilities
        """
        assert attention_mask is not None, "attention mask is none"

        bert_output = self.bert_model(input_ids=features,
                                      attention_mask=attention_mask,
                                      output_hidden_states=True)
        _e1 = bert_output['hidden_states'][-4:]

        _e2 = torch.cat((_e1[0], _e1[1], _e1[2], _e1[3]), 2)

        masked_average = self.masked_average(_e2, target_mask)
        combined_lin = self.dropout(masked_average)
        logits = self.classifier(combined_lin)

        return logits