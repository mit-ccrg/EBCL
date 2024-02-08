import dataclasses
import enum

import torch
import torch.nn.functional as F
from torch import nn

from src.model.utils import EncoderBase, OutputBase
from src.utils.train_utils import Architecture
from xformers.factory import xFormerEncoderBlock, xFormerEncoderConfig
from xformers.factory.model_factory import xFormer, xFormerConfig
from src.configs.model_configs import StratsModelConfig


class Dtype(enum.Enum):
    CAT = "cat"
    CONT = "cont"


class Triplet(enum.Enum):
    DATE = "date"
    VALUE = "value"
    VARIABLE = "variable"


architecture_to_encoder_layers = {
    Architecture.XFORMER: 1,
    Architecture.SMALL: 1,
    Architecture.MEDIUM: 2,
    Architecture.LARGE: 4,
}

architecture_to_nheads = {
    Architecture.XFORMER: 4,
    Architecture.SMALL: 2,
    Architecture.MEDIUM: 4,
    Architecture.LARGE: 8,
}


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


class FFAttention(torch.nn.Module):
    """
    Source paper: https://arxiv.org/pdf/1512.08756.pdf
    `FFAttention` is the Base Class for the Feed-Forward Attention Network.
    It is implemented as an abstract subclass of a PyTorch Module. You can
    then subclass this to create an architecture adapted to your problem.

    The FeedForward mecanism is implemented in five steps, three of
    which have to be implemented in your custom subclass:

    1. `embedding` (NotImplementedError)
    2. `activation` (NotImplementedError)
    3. `attention` (Already implemented)
    4. `context` (Already implemented)
    5. `out` (NotImplementedError)

    Attributes:
        batch_size (int): The batch size, used for resizing the tensors.
        T (int): The length of the sequence.
        D_in (int): The dimension of each element of the sequence.
        D_out (int): The dimension of the desired predicted quantity.
        hidden (int): The dimension of the hidden state.
        batch_size=args.batch_size, T=args.seq_len, D_in=args.embed_size, D_out=, hidden=None
    """

    def __init__(self, args):
        super(FFAttention, self).__init__()
        # Net Config
        self.T = args.seq_len
        self.n_features = args.embed_size
        self.out_dim = 1
        self.layer = torch.nn.Linear(args.embed_size, self.out_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def activation(self, h_t):
        """
        Step 2:
        Compute the embedding activations e_t

        In : torch.Size([batch_size, sequence_length, hidden_dimensions])
        Out: torch.Size([batch_size, sequence_length, 1])
        """
        return F.tanh(self.layer(h_t))

    def attention(self, e_t, mask):
        """
        Step 3:
        Compute the probabilities alpha_t

        In : torch.Size([batch_size, sequence_length, 1])
        Out: torch.Size([batch_size, sequence_length, 1])
        """
        alphas = self.softmax(e_t + mask.unsqueeze(-1))
        return alphas

    def context(self, alpha_t, x_t):
        """
        Step 4:
        Compute the context vector c

        In : torch.Size([batch_size, sequence_length, 1]), torch.Size([batch_size, sequence_length, sequence_dim])
        Out: torch.Size([batch_size, 1, hidden_dimensions])
        """
        batch_size = x_t.shape[0]
        return torch.bmm(alpha_t.view(batch_size, self.out_dim, self.T), x_t).squeeze(
            dim=1
        )

    def forward(self, x_e, mask=None, training=True):
        """
        Forward pass for the Feed Forward Attention network.
        """
        self.training = training
        x_a = self.activation(x_e)
        alpha = self.attention(x_a, mask)
        x_c = self.context(alpha, x_e)
        # x_o = self.out(x_c)
        return x_c, alpha


class CVE(nn.Module):
    """Continuous Value Encoder (CVE) module.

    Assumes input is a single continuous value, and encodes it
    as an `output_dim` size embedding vector.
    """

    def __init__(self, args):
        super(CVE, self).__init__()
        self.layer = nn.Linear(1, args.embed_size)

    def forward(self, x):
        return self.layer(x)


class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.
    Copied from: https://github.com/pytorch/examples/blob/main/word_language_model/model.py"""

    def __init__(self, args: StratsModelConfig):
        super(TransformerModel, self).__init__(
            d_model=args.embed_size,
            nhead=architecture_to_nheads[args.architecture_size],
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=architecture_to_encoder_layers[args.architecture_size],
            dropout=args.dropout, batch_first=args.batch_first)
        self.args = args
        self.model_type = "Transformer"
        self.src_mask = None
        self.decoder = FFAttention(args)

        self.date_embedder = CVE(args)
        self.cat_value_embedder = torch.nn.Embedding(
            args.n_cat_value, embedding_dim=args.embed_size
        )
        self.cont_value_embedder = CVE(args)
        self.variable_embedder = torch.nn.Embedding(
            args.n_variable, embedding_dim=args.embed_size
        )

    def embed_func(self, embedder, x):
        out = embedder.forward(x[None, :].T)
        return out

    def get_embedding(self, batch, dtype: Dtype):
        date, value, variable, length = (
            batch[dtype.value]["date"],
            batch[dtype.value]["value"],
            batch[dtype.value]["variable"],
            batch[dtype.value]["length"],
        )
        date_emb = self.embed_func(self.date_embedder, date)
        var_emb = self.embed_func(self.variable_embedder, variable)
        var_emb = var_emb.squeeze(dim=1)

        if dtype == Dtype.CAT:
            val_emb = self.embed_func(self.cat_value_embedder, value)
            val_emb = val_emb.squeeze(dim=1)
        else:
            val_emb = self.embed_func(self.cont_value_embedder, value)

        embedding = date_emb + val_emb + var_emb
        embedding = torch.split(embedding, length)

        return embedding

    def embed(self, batch):
        cont_embs = self.get_embedding(batch, Dtype.CONT)
        cat_embs = self.get_embedding(batch, Dtype.CAT)
        embedding = [
            torch.cat((cont_emb, cat_emb))
            for cont_emb, cat_emb in zip(cont_embs, cat_embs)
        ]
        lengths = [
            sum(ls)
            for ls in zip(
                batch["cont"]["length"],
                batch["cat"]["length"],
            )
        ]

        pad_emb = torch.nn.utils.rnn.pad_sequence(embedding, batch_first=True)
        pad_emb = torch.nn.functional.pad(
            pad_emb, (0, self.args.seq_len - pad_emb.shape[1])
        )
        return pad_emb, lengths

    def get_tokens(self, batch):
        src, lengths = self.embed(batch)
        # Transformer encoder src_key_padding_mask expects True values to be masked out
        src_pad_mask = ~sequence_mask(
            torch.tensor(lengths, device=src.device), self.args.seq_len
        )

        output = self.encoder(src, src_key_padding_mask=src_pad_mask)
        return output

    def forward(self, batch):
        src, lengths = self.embed(batch)
        # Transformer encoder src_key_padding_mask expects True values to be masked out
        src_pad_mask = ~sequence_mask(
            torch.tensor(lengths, device=src.device), self.args.seq_len
        )

        # TODO use src_key_padding_mask to mask out padding
        if not self.batch_first:
            src = src.transpose(1, 0)
        output = self.encoder(src, src_key_padding_mask=src_pad_mask)
        if not self.batch_first:
            output = output.transpose(1, 0)

        bool_output_mask = ~sequence_mask(
            torch.tensor(lengths, device=src.device), self.args.seq_len
        )
        output_mask = bool_output_mask.to(output.dtype)
        output_mask[bool_output_mask] = -torch.inf
        output, attn = self.decoder(output, output_mask)
        return output, attn
