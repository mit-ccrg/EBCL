import numpy as np
import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = args.eps

    def forward(self, yhat, y):
        if yhat.size() == y.size():
            loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        else:  # when y.size() == (batch, )
            loss = torch.sqrt(self.mse(yhat, y.unsqueeze(1)) + self.eps)
        return loss


def get_loss(args):
    if args.train_mode == "regression":
        return RMSELoss(args)
        # return nn.MSELoss()

    elif args.train_mode == "binary_class":
        # return nn.CrossEntropyLoss()
        return nn.BCEWithLogitsLoss()
