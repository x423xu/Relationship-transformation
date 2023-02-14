import importlib

losses_catalog = {"structure_loss": "StructureLoss", "value_loss": "ValueLoss"}


def make_losses(args):
    losses = []
    loss_types = args.losses
    for m in loss_types:
        loss_module = importlib.import_module("metrics.{}".format(m))
        losses.append(getattr(loss_module, losses_catalog[m])(args))
    return losses

def make_metrics(args):
    metrics = {}
    ks = args.K
    for k in ks:
        metric_module = importlib.import_module("metrics.{}".format('val_metrics'))
        metrics.update({'Recall{}'.format(k): getattr(metric_module, 'RecallK')(k, args)})
    return metrics
