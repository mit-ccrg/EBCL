import copy
import dataclasses
import logging
import os
import random
import shutil
import sys
from functools import partial
from typing import Optional

import numpy as np
import ray
import torch
import torch.optim as optim
from ray import air, tune
from ray.air import Checkpoint, session
from ray.experimental.tqdm_ray import tqdm
from ray.tune.schedulers import ASHAScheduler, pb2
from ray.tune.search.hyperopt import HyperOptSearch
from torch.utils.data import Dataset

from model.baseline_models import MLP, EncoderOutput
from model.multimodal_contrastive import CLIPOutput
from model.utils import get_baseline_model, get_model
from src.args.config import parser
from src.data.dataloaders.ebcl_dataloader import EBCLDataLoaderCreator, push_to
from src.utils.train_utils import EarlyStopping
from utils.logger import Logger, Split, TuneLogger

# os.environ["TORCH_USE_CUDA_DSA"] = "True"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
NUM_CPUS = 32


def freeze_train_model(args, train_embed, val_embed, train_loop, config):
    # Changing the model arguments from default to tune run arguments
    args = copy.deepcopy(args)
    args.lr = config["lr"]
    args.dropout = config["dropout"]
    if config["use_extra_ckpt"]:
        args.load_ckpt = args.extra_ckpts[config["seed"]]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # Log results to the Trial directory created by ray, which is the current working directory (cwd).
    logger = TuneLogger(args, save_dir=os.getcwd())

    assert torch.cuda.is_available(), "Must have CUDA available!"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up data loaders, model, optimizer, and early stopping
    model = MLP(args)
    model = model.to(device)
    train_loader = get_dataloader(args, train_embed)
    val_loader = get_dataloader(args, val_embed)
    # Early Stopping
    early_stopping = EarlyStopping(args.early_stop_tol, min_delta=0)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    # Train the model
    for epoch in tqdm(range(args.epochs), total=args.epochs, position=0, desc=f"Epoch"):
        if early_stopping.early_stop:
            break
        avg_train_loss, avg_val_loss = train_loop(
            args,
            model,
            device,
            train_loader,
            val_loader,
            optimizer,
            epoch,
            early_stopping,
            logger,
        )
        logger.log_avg_loss(avg_train_loss, Split.TRAIN)
        logger.log_avg_loss(avg_val_loss, Split.VAL)
        logger.save(model, optimizer, epoch, config=config)
        session.report(
            {
                "val_loss": avg_val_loss,
                "epoch": epoch,
                "dir": logger.dir_save,
                "best_epoch": logger.bestloss_iter,
                "best_val_loss": logger.best_loss,
            },
        )


def ray_freeze_train_model(args, train_embed_ref, val_embed_ref, train_loop, config):
    train_embed = ray.get(train_embed_ref)
    val_embed = ray.get(val_embed_ref)
    freeze_train_model(args, train_embed, val_embed, train_loop, config)


@dataclasses.dataclass
class FrozenEmbeddings:
    pre: torch.Tensor
    post: Optional[torch.Tensor]
    outcome: torch.Tensor


class FrozenDataset(Dataset):
    def __init__(self, embeds: FrozenEmbeddings):
        self.embeds = embeds

    def __len__(self):
        return self.embeds.outcome.shape[0]

    def __getitem__(self, idx):
        item = {
            "pre": self.embeds.pre[idx],
            "outcome": self.embeds.outcome[idx],
        }
        if self.embeds.post is not None:
            item["post"] = self.embeds.post[idx]
        return item


def get_dataloader(args, embeds: FrozenEmbeddings):
    dataset = FrozenDataset(embeds)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_cpus,
        pin_memory=True,
    )


def get_clip_embeddings(args, data_loader, device, model):
    assert not (
        args.unimodal and args.post
    ), "Cannot use post data for unimodal training for now."
    model.eval()
    outcomes = []
    pre_embeds = []
    post_embeds = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = push_to(batch, device)
            clip_output: CLIPOutput = model(batch)
            pre_embeds.append(clip_output.pre_embeds.detach().cpu())
            if not args.unimodal and not args.early_fusion:
                post_embeds.append(clip_output.post_embeds.detach().cpu())
            outcomes.append(batch["outcome"].detach().cpu())
            if args.debug:
                break
    pre_embeds = torch.concat(pre_embeds)
    if args.unimodal or args.early_fusion:
        post_embeds = None
    else:
        post_embeds = torch.concat(post_embeds)
    outcomes = torch.concat(outcomes)
    embeds = FrozenEmbeddings(pre=pre_embeds, post=post_embeds, outcome=outcomes)
    return embeds


def get_ocp_embeddings(data_loader, device, model):
    model.eval()
    outcomes = []
    pre_embeds = []
    post_embeds = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = push_to(batch, device)
            clip_output: CLIPOutput = model(batch)
            pre_embeds.append(clip_output.pre_embeds.detach().cpu())
            post_embeds.append(clip_output.post_embeds.detach().cpu())
            outcomes.append(batch["outcome"].detach().cpu())
    pre_embeds = torch.concat(pre_embeds)
    post_embeds = torch.concat(post_embeds)
    outcomes = torch.concat(outcomes)
    embeds = FrozenEmbeddings(pre=pre_embeds, post=post_embeds, outcome=outcomes)
    return embeds
