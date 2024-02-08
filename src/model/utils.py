import dataclasses

import torch
import torchvision
from torch import nn


@dataclasses.dataclass
class OutputBase:
    loss: torch.Tensor


class EncoderBase(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, batch) -> OutputBase:
        raise NotImplementedError


def get_contrastive_embeddings(tmodel, batch1, batch2, device):
    # extract feature representations of each modality
    I_f = tmodel.encoder1(batch1).double().to(device)  # [n, d_i]
    T_f = tmodel.encoder2(batch2).double().to(device)  # [n, d_t]
    # joint multimodal embedding [n, d_e]
    I_e = torch.nn.functional.normalize(tmodel.data1_projection(I_f), p=2, dim=1)
    T_e = torch.nn.functional.normalize(tmodel.data2_projection(T_f), p=2, dim=1)

    return I_e, T_e


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args

        initial_projection_dim = 2 * args.projection_dim
        if args.unimodal or args.early_fusion:
            initial_projection_dim = args.projection_dim

        self.mlp = torchvision.ops.MLP(
            in_channels=initial_projection_dim,
            hidden_channels=[args.projection_dim, args.projection_dim, 1],
            activation_layer=torch.nn.ReLU,
            norm_layer=torch.nn.BatchNorm1d,
            dropout=args.dropout,
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, batch):
        return self.mlp(batch)
