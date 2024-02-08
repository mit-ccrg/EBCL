"""This is expected to be run in a Jupyter notebook.
Helper functions are provided her for getting clip embeddings and visualizing them.
"""
import dataclasses
import enum

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm

from data.time_data_loader import push_to
from model.multimodal_contrastive import CLIPOutput, Modality


class EmbeddingType(enum.Enum):
    TRANSFORMER_TOKENS = "TRANSFORMER_TOKENS"
    TRANSFORMER_OUTPUT = "TRANSFORMER_OUTPUT"
    PROJECTION = "PROJECTION"
    NORMALIZED_PROJECTION = "NORMALIZED_PROJECTION"


@dataclasses.dataclass
class Embeddings:
    pre_embeds: np.ndarray
    post_embeds: np.ndarray
    outcomes: np.ndarray
    distances: np.ndarray
    outcome_name: str
    processed_outcomes: np.ndarray
    threshold: float = None
    type: EmbeddingType = None


@dataclasses.dataclass
class FullEmbeddings:
    pre_tokens: np.ndarray
    post_tokens: np.ndarray

    pre_model_output: np.ndarray
    post_model_output: np.ndarray

    pre_projected_embeds: np.ndarray
    post_projected_embeds: np.ndarray

    pre_normalized_embeds: np.ndarray
    post_normalized_embeds: np.ndarray

    outcomes: np.ndarray
    distances: np.ndarray
    outcome_name: str
    processed_outcomes: np.ndarray = None
    threshold: float = None

    def get_specific_embeddings(self, embedding_type: EmbeddingType):
        if embedding_type == EmbeddingType.TRANSFORMER_TOKENS:
            pre_embed = self.pre_tokens
            post_embed = self.post_tokens
        elif embedding_type == EmbeddingType.TRANSFORMER_OUTPUT:
            pre_embed = self.pre_model_output
            post_embed = self.post_model_output
        elif embedding_type == EmbeddingType.PROJECTION:
            pre_embed = self.pre_projected_embeds
            post_embed = self.post_projected_embeds
        elif embedding_type == EmbeddingType.NORMALIZED_PROJECTION:
            pre_embed = self.pre_normalized_embeds
            post_embed = self.post_normalized_embeds
        else:
            raise ValueError(f"Invalid embedding type: {embedding_type}")
        return Embeddings(
            pre_embed,
            post_embed,
            self.outcomes,
            self.distances,
            self.outcome_name,
            self.processed_outcomes,
            self.threshold,
            type=embedding_type,
        )


def get_embeddings(
    test_loader, device, model, outcome_name: str, threshold: float = None
):
    pre_tokens, post_tokens = [], []
    pre_model_output, post_model_output = [], []
    pre_projected_embeds, post_projected_embeds = [], []
    pre_normalized_embeds, post_normalized_embeds = [], []
    processed_outcomes = []
    outcomes = []
    distances = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = push_to(batch, device)
            clip_output: CLIPOutput = model(batch)

            pre_tokens.append(
                model.pre_model.get_tokens(batch, Modality.PRE).detach().cpu().numpy()
            )
            post_tokens.append(
                model.post_model.get_tokens(batch, Modality.POST).detach().cpu().numpy()
            )

            pre_model_output.append(clip_output.pre_model_output.detach().cpu().numpy())
            post_model_output.append(
                clip_output.post_model_output.detach().cpu().numpy()
            )

            pre_projected_embeds.append(clip_output.pre_embeds.detach().cpu().numpy())
            post_projected_embeds.append(clip_output.post_embeds.detach().cpu().numpy())

            pre_normalized_embeds.append(
                clip_output.pre_norm_embeds.detach().cpu().numpy()
            )
            post_normalized_embeds.append(
                clip_output.post_norm_embeds.detach().cpu().numpy()
            )

            distances.append(
                torch.mm(clip_output.post_norm_embeds, clip_output.pre_norm_embeds.T)
                .diagonal()
                .detach()
                .cpu()
                .numpy()
            )
            outcomes.append(batch["raw_outcome"].detach().cpu().numpy())
            processed_outcomes.append(batch["outcome"].detach().cpu().numpy())

    pre_tokens = np.concatenate(pre_tokens)
    post_tokens = np.concatenate(post_tokens)

    pre_model_output = np.concatenate(pre_model_output)
    post_model_output = np.concatenate(post_model_output)

    pre_projected_embeds = np.concatenate(pre_projected_embeds)
    post_projected_embeds = np.concatenate(post_projected_embeds)

    pre_normalized_embeds = np.concatenate(pre_normalized_embeds)
    post_normalized_embeds = np.concatenate(post_normalized_embeds)

    outcomes_array = np.concatenate(outcomes)
    processed_outcomes_array = np.concatenate(processed_outcomes)
    distances_array = np.concatenate(distances)
    embeddings = FullEmbeddings(
        pre_tokens,
        post_tokens,
        pre_model_output,
        post_model_output,
        pre_projected_embeds,
        post_projected_embeds,
        pre_normalized_embeds,
        post_normalized_embeds,
        outcomes_array,
        distances_array,
        outcome_name,
        processed_outcomes=processed_outcomes_array,
        threshold=threshold,
    )
    return embeddings


def distance_correlation_plot(embeddings: Embeddings):
    outcomes = embeddings.outcomes
    distances = embeddings.distances
    outcome_name = embeddings.outcome_name
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5))
    color = mpl.colormaps["viridis"](0.1)
    line_color = mpl.colormaps["viridis"](0.6)
    df = pd.DataFrame({"distances": distances, outcome_name: outcomes})
    sns.regplot(
        data=df,
        x="distances",
        scatter_kws={"alpha": 0.5, "color": color},
        y=outcome_name,
        line_kws={"color": line_color},
        ci=None,
        ax=ax,
    )
    ax.set_ylabel(f"log {outcome_name}")
    ax.set_yscale("log")
    corr, pval = scipy.stats.pearsonr(distances, outcomes)
    ax.set_title(
        f"Distance vs {outcome_name}. (Pearsonr, pval): {corr:.3f}, {pval:.3f}"
    )


def plot_reduced_embeddings(pre_ax, post_ax, reduced_embedding: Embeddings):
    no_survive_color = mpl.colormaps["viridis"](0.1)
    survive_color = mpl.colormaps["viridis"](0.6)

    def plot_embeddings(ax, embed, survived, title):
        ax.scatter(
            embed[survived][:, 0],
            embed[survived][:, 1],
            label=f"{reduced_embedding.threshold} day {reduced_embedding.outcome_name}",
            color=survive_color,
            alpha=0.25,
        )
        ax.scatter(
            embed[~survived][:, 0],
            embed[~survived][:, 1],
            label=f"No-{reduced_embedding.outcome_name}",
            color=no_survive_color,
            alpha=0.25,
        )
        ax.legend(loc="upper right")
        ax.set_title(title)
        ax.set_xlabel("DIM_1")
        ax.set_ylabel("DIM_2")

    survived = reduced_embedding.processed_outcomes
    plot_embeddings(pre_ax, reduced_embedding.pre_embeds, survived, "Pre Embedding")
    plot_embeddings(post_ax, reduced_embedding.post_embeds, survived, "Post Embedding")


def dim_reduction_plot(dim_reduction, embeddings: Embeddings):
    assert dim_reduction in ["pca", "tsne"]
    if dim_reduction == "pca":
        reduce_pre_embed = PCA(n_components=2).fit_transform(embeddings.pre_embeds)
        reduce_post_embed = PCA(n_components=2).fit_transform(embeddings.post_embeds)
    elif dim_reduction == "tsne":
        reduce_pre_embed = TSNE(n_components=2).fit_transform(embeddings.pre_embeds)
        reduce_post_embed = TSNE(n_components=2).fit_transform(embeddings.post_embeds)
    reduced_embedding = Embeddings(
        reduce_pre_embed,
        reduce_post_embed,
        embeddings.outcomes,
        embeddings.distances,
        embeddings.outcome_name,
        threshold=embeddings.threshold,
        processed_outcomes=embeddings.processed_outcomes,
    )
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    pre_ax, post_ax = ax

    plot_reduced_embeddings(pre_ax, post_ax, reduced_embedding)
