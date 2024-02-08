import numpy as np
import torch
from tqdm.auto import tqdm

from src.configs.model_configs import DuettModelConfig
from src.configs.train_configs import BaseTrainConfig, SupervisedTrainConfig
from src.data.dataloaders.ebcl_dataloader import push_to
from src.model.duett_model import DuettModule
from src.utils.logger import Logger, Split, TuneLogger


def forward_pass(args, model, batch, device):
    """Pushes batch to device, and performs a forward pass on the model."""
    batch = push_to(batch, device)
    output = model(batch)
    return output


def get_loss(args: BaseTrainConfig, model, batch, device):
    """Pushes batch to device, and performs a forward pass on the model."""
    model_config = args.model_config
    if isinstance(model, DuettModule):
        assert isinstance(model_config, DuettModelConfig)
        assert isinstance(model, DuettModule)
        loss = model.training_step(batch, batch_idx=None)
    else:
        output = forward_pass(model_config, model, batch, device)
        loss = output.loss
    assert not torch.isnan(loss), f"Train Loss is {loss.item()}"
    return loss


def get_val_loss(args: BaseTrainConfig, model, batch, device):
    """Pushes batch to device, and performs a forward pass on the model."""
    model_config = args.model_config
    if isinstance(model, DuettModule):
        assert isinstance(model_config, DuettModelConfig)
        assert isinstance(model, DuettModule)
        model.validation_step(batch, batch_idx=None)
        loss = model.val_loss.compute()
        import pdb

        pdb.set_trace()
    else:
        output = forward_pass(model_config, model, batch, device)
        loss = output.loss
    assert not torch.isnan(loss), f"Train Loss is {loss.item()}"
    return loss


def val_loop(args, model, device, val_loader, logger: TuneLogger):
    """Runs a validation loop over the validation set, and logs the results."""
    model.eval()
    max_val_iter = args.max_val_iter
    if max_val_iter is None:
        max_val_iter = len(val_loader)

    loss_iter = []
    with torch.no_grad():
        for batch in tqdm(
            val_loader, position=2, total=max_val_iter, desc=f"VAL loader progress"
        ):
            loss = get_val_loss(args, model, batch, device)
            assert not torch.isnan(loss), f"Val Loss is {loss.item()}"
            loss_iter.append(loss.item())
            logger.log_batch_loss(loss.item(), Split.VAL)
            if args.debug:
                break

        loss_iter = np.array(loss_iter)
        loss_iter = loss_iter.squeeze()

    model.train()  # Set the model back to training mode
    return loss_iter.mean()


def train_loop(
    args,
    model,
    device,
    train_loader,
    val_loader,
    optimizer,
    epoch,
    early_stopping,
    logger: TuneLogger,
):
    model.train()
    loss_iter = []

    for i, batch in enumerate(
        tqdm(train_loader, position=1, desc=f"Epoch {epoch}, train loader progress")
    ):
        loss = get_loss(args, model, batch, device)
        assert not torch.isnan(loss), f"Train Loss is {loss.item()}"

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # LOGGING Train and Validation set performance
        loss_iter.append(loss.item())
        logger.log_batch_loss(loss.item(), Split.TRAIN)
        if args.debug:
            break
    avg_val_loss = val_loop(args, model, device, val_loader, logger)
    early_stopping(avg_val_loss)

    return np.mean(loss_iter), avg_val_loss


def test_model(model, args, test_loader, device):
    assert isinstance(
        args, SupervisedTrainConfig
    ), "Model testing is only for supervised models"
    model.eval()
    logger = Logger(args, model)
    logger.evaluator.reset()
    logger.loss_reset()
    with torch.no_grad():
        loss_iter = []
        for batch in tqdm(test_loader, position=1, desc=f"Test Loader Progress"):
            batch = push_to(batch, device)
            final_output, loss = forward_pass(args, model, batch, device)
            test_y = batch["outcome"].float()

            logger.loss += loss.item()
            loss_iter.append(loss.item())
            logger.evaluator.add_task_batch(
                test_y.cpu().numpy(),
                torch.sigmoid(final_output.float()).data.cpu().numpy(),
                loss.item(),
            )

        loss_iter = np.array(loss_iter)

    if args.train_mode == "binary_class":
        f1, auc, apr, acc = logger.evaluator.performance_task()
        print("f1: {}, auc: {}, apr: {}, acc: {}".format(f1, auc, apr, acc))
        result_dict = {"auc": auc, "apr": apr, "acc": acc, "f1": f1}
    elif args.train_mode == "regression":
        loss, r, pval = logger.evaluator.performance_task()
        print("loss: {}, pearsonr: {}, pval : {}".format(loss, r, pval))
        result_dict = {"rmse": loss}

    return result_dict
