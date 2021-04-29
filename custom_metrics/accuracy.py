import torch


def bin_acc(y_true, y_pred, sigmoid: bool = False):
    """
    Returns accuracy per batch (with sigmoid function), i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    if sigmoid:
        round_pred = torch.round(torch.sigmoid(y_pred))
    else:
        round_pred = torch.round(y_pred)
    correct = (round_pred == y_true).float()
    acc = correct.sum() / len(correct)
    return acc