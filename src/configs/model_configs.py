from src.configs.dataloader_configs import SupervisedView
from src.configs.utils import hydra_dataclass
from src.utils.train_utils import Architecture


@hydra_dataclass
class StratsModelConfig:
    """EBCL Configuration

    Attributes:
        model: Model architecture to use
        seq_len: Max sequence length to use for transformer encoder
        embed_size: Embedding size for input `tokens` to transformer
        dim_feedforward: Dimension of the feedforward network for transformer encoder
        dropout: Dropout probability for model
        projection_dim: Final output projection dimension for Model
        encoder_dim: Final output `token` dimension for encoder
        architecture_size: Size of transformer encoder overrides number of attention heads and encoders.
        n_cat_value: Number of unique categorical values in the dataset
        n_variable: Number of unique variables in the dataset
        eps: eps for RMSE
    """
    pretrain: bool = True
    seq_len: int = 512
    embed_size: int = 32
    dim_feedforward: int = 128
    dropout: float = 0.5
    projection_dim: int = 32
    encoder_dim: int = 32
    architecture_size: Architecture = Architecture.MEDIUM
    n_cat_value: int = 7761
    n_variable: int = 3275
    n_cat_variable: int = 52 # variables 52 and up are continuous
    eps: float = 1e-4
    modalities: SupervisedView = SupervisedView.PRE_AND_POST
    lr: float = 3.0e-4
    shared_transformer: bool = True
    batch_size: int = 256
    half_dtype: bool = False
    batch_first: bool = True
    bootstrap: bool = False

    def __post_init__(self):
        assert self.embed_size == self.projection_dim
        assert self.embed_size == self.encoder_dim


@hydra_dataclass
class DuettModelConfig:
    """Duett Model Configuration

    Attributes:
        none
    """

    lr: float = 3.0e-4  # for finetuning should be around 1.e-5
    dropout: float = 0.5
    seq_len: int = 32  # number of time bins
    num_event: int = 32  # number of event types
    # architecture_size: Architecture = Architecture.MEDIUM
    d_target: int = (
        1  # number of classes for target, should be 1 as we do binary classification
    )
    pretrain: bool = True  # whether to pretrain the model
    batch_size: int = 256
    # embed_size: int = 64  # Embedding dimension outside Duett Transformer
    # d_embedding: int = 24  # embedding dimension within Duett Transformer
    half_dtype: bool = False
