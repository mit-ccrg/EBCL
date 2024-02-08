import logging
import os

import torch

from .baseline_models import SupervisedEarlyFusionModule, TimeTabularSharedEncoderModel
from .ebcl_model import PretrainEBCLModule
from .ocp_model import PretrainOCPModule
from .utils import MLP


def get_model(args, device):
    if args.model == "clip":
        model = PretrainEBCLModule(args)
    elif args.model == "ocp":
        model = PretrainOCPModule(args)
    else:
        raise NotImplementedError(f"model {args.model} not implemented")

    if args.finetune:
        assert args.load_ckpt is not None, "Must specify load step for finetuning."

        if not os.path.exists(args.load_ckpt):
            print("invalid checkpoint path : {}".format(args.load_ckpt))

        logging.info("Loading last checkpoint from {}".format(args.load_ckpt))
        ckpt = torch.load(args.load_ckpt, map_location=device)
        state = ckpt["model"]
        model.load_state_dict(state)
    elif args.load_ckpt is not None:
        logging.info("Loading last checkpoint from {}".format(args.load_ckpt))
        ckpt = torch.load(args.load_ckpt, map_location=device)
        state = ckpt["model"]
        model.load_state_dict(state)

    model.to(device)
    return model


def get_frozen_model(args, device):
    model = MLP(args)
    assert args.load_ckpt is not None, "Must specify load step for finetuning."
    ckpt = torch.load(args.load_ckpt, map_location=device)
    state = ckpt["model"]
    model.load_state_dict(state)
    model.to(device)
    return model


def get_baseline_model(args, device):
    """Get a baseline model for our finetuning tasks.

    if load_ckpt is specified, then load transformer from that checkpoint and randomly intializes other weights
    if load_ckpt is not specified, then randomly initialize all weights
    """
    if args.early_fusion:
        model = SupervisedEarlyFusionModule(args)
    else:
        model = TimeTabularSharedEncoderModel(args)

    if args.load_ckpt:
        if args.model == "clip":
            clip_model = get_model(args, device)
            assert isinstance(clip_model, PretrainEBCLModule)
            model.pre_model = clip_model.pre_model
            model.post_model = clip_model.post_model
            model.post_projection = clip_model.post_projection
            model.pre_projection = clip_model.pre_projection
            if args.early_fusion:
                raise NotImplementedError(
                    "early fusion works for clip, but this isn't an experiment we are doing so remove the flag"
                )
        elif args.model == "ocp":
            ocp_model = get_model(args, device)
            assert isinstance(ocp_model, PretrainOCPModule)
            if args.early_fusion:
                model.transformer = ocp_model.transformer
            else:
                model.pre_model = ocp_model.transformer
                model.post_model = ocp_model.transformer
        else:
            raise NotImplementedError(f"model {args.model} not implemented")
    model.to(device)
    return model


def get_finetuned_model(args, device):
    """Get a baseline model for our finetuning tasks.

    if load_ckpt is specified, then load ALL weights from load_ckpt path
    if load_ckpt is not specified, then randomly initialize all weights
    """
    if args.early_fusion:
        model = SupervisedEarlyFusionModule(args)
    else:
        model = TimeTabularSharedEncoderModel(args)

    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device)
        state = ckpt["model"]
        model.load_state_dict(state)
    model.to(device)
    return model
