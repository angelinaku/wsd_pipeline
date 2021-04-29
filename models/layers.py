import torch
from torch import nn


class SpatialDropout(nn.Module):
    """
    Spatial Dropout drops a certain percentage of dimensions from each word vector in the training sample
    implementation: https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400
    explanation: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/76883
    """

    def __init__(self, p: float):
        super(SpatialDropout, self).__init__()
        self.spatial_dropout = nn.Dropout2d(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # convert to [batch, channels, time]
        x = self.spatial_dropout(x)
        x = x.permute(0, 2, 1)  # back to [batch, time, channels]
        return x


class MaskedAverageLayer(nn.Module):
    def __init__(self):
        super(MaskedAverageLayer, self).__init__()

    def forward(self, seq, mask):
        '''
        Inputs:
            -seq : Tensor of shape [B, T, E] containing embeddings of sequences
            -mask : Tensor of shape [B, T, 1] containing masks to be used to pull from seq
        '''
        output = None
        if mask is not None:
            if len(mask.shape) < len(seq.shape):
                mask = mask.unsqueeze(-1)
                mask = mask.repeat(1, 1, seq.shape[-1])

            masked_inputs = (mask.int()*seq) + (1-mask.int())*torch.zeros_like(seq)
            unmasked_counts = torch.sum(mask.float(), dim=1)
            output = torch.sum(masked_inputs, dim=1)/(unmasked_counts+1e-10)

        return output

