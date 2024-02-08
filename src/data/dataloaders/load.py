from .ebcl_dataloader import EBCLDataLoaderCreator
from .ocp_dataloader import OCPDataLoaderCreator
from .utils import BaseDataloaderCreator


def load_dataloader_creator(args, device) -> BaseDataloaderCreator:
    if args.dataset == "ebcl":
        return EBCLDataLoaderCreator(args, device)
    elif args.dataset == "ocp":
        return OCPDataLoaderCreator(args, device)
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")
