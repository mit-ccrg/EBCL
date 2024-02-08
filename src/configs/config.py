import argparse
import enum

from src.utils.data_utils import EncounterSet
from src.utils.process_utils import file_path, file_path_list
from src.utils.train_utils import Architecture

parser = argparse.ArgumentParser()


# General Parameters
parser.add_argument(
    "--debug",
    default=False,
    action="store_true",
    help="debug mode - run one batch and 4 epochs",
)
parser.add_argument("--name", type=str, default="test")
parser.add_argument("--gpu", type=int)
parser.add_argument("--reset", default=False, action="store_true")
parser.add_argument("--data_seed", type=int, default=926)
parser.add_argument("--pretrain_seed", type=int, default=1)
parser.add_argument("--finetune_seed", type=int, default=2)
parser.add_argument(
    "--inpatient-only",
    default=False,
    action="store_true",
    help="Only generate data for inpatient encounters.",
)

# Training Parameters
parser.add_argument(
    "--early-fusion",
    default=False,
    action="store_true",
    help="Fuse pre and post before passing to transformer. Effectively halves sequence length as pre+post sequence can be amx of 512 tokens.",
)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument(
    "--early-stop-tol",
    type=int,
    default=3,
    help="Number of val-iters to wait before early stopping.",
)
parser.add_argument(
    "--unimodal", default=False, action="store_true", help="Unimodal Training."
)
parser.add_argument(
    "--separate_encoder",
    default=False,
    action="store_true",
    help="Separate encoder for pre/post data.",
)
parser.add_argument(
    "--post",
    default=False,
    action="store_true",
    help="Use Post vs. Post for unimodal contrastive learning, Pre data is default.",
)
parser.add_argument(
    "--finetune",
    default=False,
    action="store_true",
    help="Finetune the contrastive model with downstream outcome prediction task.",
)
parser.add_argument(
    "--freeze",
    default=False,
    action="store_true",
    help="Freeze embeddings for downstream task prediction.",
)

# Data Parameters
parser.add_argument("--data-dir", type=str, default="./data/")
parser.add_argument(
    "--max-obs",
    type=int,
    default=1024,
    help="max number of observations per encounter for pre and post data, subsample this number if more than this number",
)
parser.add_argument(
    "--min-obs",
    type=int,
    default=100,
    help="minimum number of pre and post observations per encounter, drop encounter if less than this number",
)
parser.add_argument(
    "--num-cpus", type=int, default=8, help="number of cpus to use for multiprocessing"
)
parser.add_argument(
    "--post-cutoff",
    type=int,
    default=None,
    help="minimum number of days between encounters",
)
parser.add_argument(
    "--post-los-cutoff",
    default=False,
    action="store_true",
    help="Cutoff post data at LOS + 1 day.",
)
parser.add_argument(
    "--pre-cutoff",
    type=int,
    default=None,
    help="minimum number of days between encounters",
)
parser.add_argument(
    "--encounter-set",
    type=EncounterSet,
    choices=list(EncounterSet),
    help="Encounters to generate data for.",
    required=True,
)
parser.add_argument("--discharge-event", default=False, action="store_true")
parser.add_argument(
    "--subsample",
    type=float,
    default=1,
    help="subsampling percentage for fewshot finetuning",
)
parser.add_argument(
    "--class_imbalance",
    type=str,
    default="default",
    help="specifies how much class imbalance is desired in train/val",
)

# Model Parameters
parser.add_argument("--model", type=str, choices=["clip", "ocp"])
parser.add_argument("--seq-len", type=int, default=512)
parser.add_argument(
    "--embed-size", type=int, default=32, help="Embedding size for input `tokens`"
)
parser.add_argument(
    "--dim-feedforward",
    type=int,
    default=2048,
    help="Dimension of the feedforward network for transformer encoder",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.5,
    help="Dropout probability for transformer encoder",
)
parser.add_argument(
    "--projection-dim",
    type=int,
    default=32,
    help="Final output projection dimension for CLIP",
)
parser.add_argument(
    "--encoder-dim",
    type=int,
    default=32,
    help="Final output `token` dimension for encoder",
)
parser.add_argument(
    "--architecture",
    type=Architecture,
    choices=list(Architecture),
    default=Architecture.MEDIUM,
    help="Deciding size of transformer, overrides number of attention heads and encoders.",
)
# TODO regenerate dataset with this filtered correctly to a lower number
parser.add_argument(
    "--n-cat-value",
    type=int,
    default=7761,
    help="Number of unique categorical values in the dataset",
)
parser.add_argument(
    "--n-variable",
    type=int,
    default=3275,
    help="Number of unique variables in the dataset",
)
parser.add_argument("--eps", type=float, default=1e-6)  # eps for RMSE
# parser.add_argument("--use-identity-encoder", default=False, action="store_true", help='Use identity operation as encoder.')

# Finetuning Parameters
parser.add_argument("--best-auc", default=False, action="store_true")
parser.add_argument("--best-loss", default=False, action="store_true")
parser.add_argument("--last", default=False, action="store_true")
parser.add_argument(
    "--bootstrap", default=False, action="store_true", help="For Bootstrapping"
)
parser.add_argument(
    "--train-mode",
    type=str,
    default="binary_class",
    choices=["regression", "binary_class"],
)

#### Do not use this - it is legacy
parser.add_argument(
    "--finetune-hidden-dim",
    type=int,
    default=4,
    help="Finetuning hidden dimension size.",
)

# Logging Parameters
parser.add_argument("--val-iter", type=int, default=1)
parser.add_argument(
    "--max-val-iter",
    type=int,
    default=None,
    help="Maximum number of val iterations to run.",
)
parser.add_argument("--log-iter", type=int, default=1)
parser.add_argument(
    "--train-iter-n",
    type=int,
    default=1,
    help="Number of batches to average training metrics over (performed once each val-iter) before logging.",
)
parser.add_argument("--save-iter", type=int, default=10)
parser.add_argument("--save-dir", type=str, default="./result")

# Evaluation Parameters
parser.add_argument("--load-ckpt", type=file_path, default=None)
parser.add_argument("--extra-ckpts", type=file_path_list, default=None)
parser.add_argument(
    "--distance-met", type=str, default="cosine", choices=["cosine", "euclidean"]
)
parser.add_argument("--standardize", type=int, default=1)
parser.add_argument(
    "--synthetic-eval",
    default=False,
    action="store_true",
    help="Synthetic Data for Evaluation",
)
parser.add_argument(
    "--get-embedding",
    default=False,
    action="store_true",
    help="Get embeddings for evaluation",
)
