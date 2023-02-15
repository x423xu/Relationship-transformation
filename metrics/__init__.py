import importlib
import torch.nn as nn

losses_catalog = {"structure_loss": "StructureLoss", "value_loss": "ValueLoss"}


def make_losses(args):
    losses = []
    weights = []
    if args.simple_loss:
        loss_module = importlib.import_module("metrics.value_loss")
        losses.append(getattr(loss_module, "SimpleLoss")(args))
        weights.append(1)
    else:
        loss_types = args.losses
        for m in loss_types:
            loss_module = importlib.import_module("metrics.{}".format(m))
            losses.append(getattr(loss_module, losses_catalog[m])(args))
        weights = args.losses_weights
    return losses, weights


def make_metrics(args):
    metrics = {}
    ks = args.K
    for k in ks:
        metric_module = importlib.import_module("metrics.{}".format("val_metrics"))
        metrics.update(
            {"Recall{}".format(k): getattr(metric_module, "RecallK")(k, args)}
        )
    metrics.update({"Precision": getattr(metric_module, "Precision")()})
    return metrics
