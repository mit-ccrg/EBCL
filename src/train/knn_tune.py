# Loads CLIP Model, make sure to edit arg_string to point to the correct checkpoint and have your model arguments
import argparse
import dataclasses
import enum
import logging
import os
import sys
from functools import partial
from typing import Sequence

import numpy as np
import ray
import torch
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from model.multimodal_contrastive import CLIPOutput
from model.plot import get_embeddings
from model.utils import get_model
from src.args.config import parser
from src.data.dataloaders.ebcl_dataloader import EBCLDataLoaderCreator, push_to

script_parser = argparse.ArgumentParser()
script_parser.add_argument(
    "--encounter-set", type=str, choices=["readmission", "mortality"], required=True
)
script_parser.add_argument("--num-samples", type=int, default=100)

# General Parameters


class Metric(enum.Enum):
    PRE_COSINE = "PRE_COSINE"
    POST_COSINE = "POST_COSINE"
    COSINE = "COSINE"


@dataclasses.dataclass
class MetricDataProcessor:
    metric: Metric

    def get_data(self, embeddings):
        pre = embeddings.pre_projected_embeds.T / np.linalg.norm(
            embeddings.pre_projected_embeds, axis=1
        )
        pre = pre.T
        post = embeddings.post_projected_embeds.T / np.linalg.norm(
            embeddings.post_projected_embeds, axis=1
        )
        post = post.T
        outcomes = embeddings.processed_outcomes
        if self.metric == Metric.PRE_COSINE:
            return pre, outcomes
        elif self.metric == Metric.POST_COSINE:
            return post, outcomes
        elif self.metric == Metric.COSINE:
            return np.hstack([pre, post]), outcomes
        else:
            raise ValueError(f"Metric {self.metric} not supported")


def get_projected_result(
    embeddings, val_embeddings, test_embeddings, encounter_set, seed
):
    concat_embeds = np.hstack(
        [embeddings.pre_projected_embeds, embeddings.post_projected_embeds]
    )
    outcomes = embeddings.processed_outcomes
    clf = KNeighborsClassifier(n_neighbors=5).fit(concat_embeds, outcomes)

    pre_tokens = val_embeddings.pre_projected_embeds
    post_tokens = val_embeddings.post_projected_embeds
    concat_embeds = np.hstack([pre_tokens, post_tokens])
    pred = clf.predict(concat_embeds)
    auc = roc_auc_score(val_embeddings.processed_outcomes, pred)
    apr = average_precision_score(val_embeddings.processed_outcomes, pred)
    accuracy = accuracy_score(val_embeddings.processed_outcomes, pred)
    val_result = dict(
        auc=auc,
        apr=apr,
        acc=accuracy,
        task=encounter_set,
        seed=seed,
        type="projected",
        split="val",
    )

    pre_tokens = test_embeddings.pre_projected_embeds
    post_tokens = test_embeddings.post_projected_embeds
    concat_embeds = np.hstack([pre_tokens, post_tokens])
    pred = clf.predict(concat_embeds)
    auc = roc_auc_score(test_embeddings.processed_outcomes, pred)
    apr = average_precision_score(test_embeddings.processed_outcomes, pred)
    accuracy = accuracy_score(test_embeddings.processed_outcomes, pred)
    test_result = dict(
        auc=auc,
        apr=apr,
        acc=accuracy,
        task=encounter_set,
        seed=seed,
        type="projected",
        split="test",
    )
    return [val_result, test_result]


def load_embeds(train_loader, val_loader, test_loader, device, model, encounter_set):
    print("Getting embeddings")
    embeddings = get_embeddings(train_loader, device, model, encounter_set)
    print("Getting val embeddings")
    val_embeddings = get_embeddings(val_loader, device, model, encounter_set)
    print("Getting test embeddings")
    test_embeddings = get_embeddings(test_loader, device, model, encounter_set)
    return embeddings, val_embeddings, test_embeddings


def get_all_embeddings(seeds: Sequence[int], encounter_set: str):
    datasets = []
    for seed in seeds:
        arg_string = f"--load-ckpt /storage/x-x-x/hf_causal/result/los_sufficient_lr_001_seed_{seed}_small/ckpts/bestloss.pth --data-dir x-x-xhf_cohort/final/ --epochs 100 --embed-size 32 --val-iter 1000 --log-iter 1 --train-iter-n 10 --num-cpus 16 --batch-size 256 --encoder-dim 32 --nhead 4 --num-encoder-layers 2 --min-obs 16 --max-obs 512 --seq-len 512 --lr .001 --seed 0 --inpatient-only --dropout 0.2 --name test --projection-dim 32 --post-los-cutoff --finetune --gpu 2 --encounter-set {encounter_set} --task {encounter_set} --train-mode binary_class"
        args = parser.parse_args(arg_string.split())

        # set up logging, and set device and seed
        logging.basicConfig(level=logging.INFO)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda")
        torch.manual_seed(args.seed)

        # Set up data loaders, model, optimizer, and early stopping
        data_loader_creator = EBCLDataLoaderCreator(args, device)
        train_loader, val_loader, test_loader = data_loader_creator.get_dataloaders()

        model = get_model(args, device)
        model.eval()
        embeddings, val_embeddings, test_embeddings = load_embeds(
            train_loader, val_loader, test_loader, device, model, encounter_set
        )
        datasets.append(
            dict(
                embeddings=embeddings,
                val_embeddings=val_embeddings,
                test_embeddings=test_embeddings,
                seed=seed,
                encounter_set=encounter_set,
            )
        )
    return datasets


def get_knn_auc(datasets, config):
    auc_scores = []
    # Unpack hyperparameters from the config parameter
    n_neighbors = config["n_neighbors"]
    metric = config["metric"]
    weights = config["weights"]
    distance = config["distance"]
    for dataset in datasets:
        embeddings = dataset["embeddings"]
        val_embeddings = dataset["val_embeddings"]

        if distance == "l1":
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, p=1, weights=weights)
        elif distance == "l2":
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, p=2, weights=weights)
        else:
            clf = KNeighborsClassifier(
                n_neighbors=n_neighbors, metric=distance, weights=weights
            )
        mdp = MetricDataProcessor(metric)
        concat_embeds, outcomes = mdp.get_data(embeddings)
        val_concat_embeds, val_outcomes = mdp.get_data(val_embeddings)

        # Create and train KNeighborsClassifier
        clf.fit(concat_embeds, outcomes)

        # Make predictions on validation set
        pred = clf.predict(val_concat_embeds)

        # Calculate AUC
        auc = roc_auc_score(val_outcomes, pred)
        auc_scores.append(auc)
    auc_avg = np.mean(auc_scores)
    # Report AUC score to Ray Tune
    return dict(auc=auc_avg)


def tune_knn(datasets_ref, config):
    datasets = ray.get(datasets_ref)
    return get_knn_auc(datasets, config)


if __name__ == "__main__":
    script_args = script_parser.parse_args()
    seeds = [0, 1, 2]
    datasets = get_all_embeddings(seeds, script_args.encounter_set)
    # Define the hyperparameter search space
    config = {
        "n_neighbors": tune.randint(1, 30),
        "metric": tune.choice([Metric.PRE_COSINE, Metric.POST_COSINE, Metric.COSINE]),
        "weights": tune.choice(["uniform", "distance"]),
        "distance": tune.choice(["l1", "l2", "cosine"]),
    }
    # test_config = {"n_neighbors": 5, "metric": Metric.COSINE, "weights": 'uniform', "distance": 'cosine'}
    # get_knn_auc(datasets, test_config)
    hyperopt_search = HyperOptSearch(metric="auc", mode="max")
    tune_config = tune.TuneConfig(
        metric="auc",
        mode="max",
        num_samples=script_args.num_samples,
        search_alg=hyperopt_search,
        max_concurrent_trials=16,
    )
    datasets_ref = ray.put(datasets)
    tuner = tune.Tuner(
        partial(tune_knn, datasets_ref), param_space=config, tune_config=tune_config
    )
    # Run hyperparameter tuning
    results = tuner.fit()
    results.get_dataframe().to_pickle(f"{script_args.encounter_set}_knn_result.pkl")
