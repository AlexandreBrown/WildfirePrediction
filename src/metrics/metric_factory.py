from metrics.pr_auc import PrecisionRecallAucMetric
from metrics.loss_metric import LossMetric
from losses.loss_factory import create_loss


def create_metrics(device, metrics_config: list, target_no_data_value: int) -> list:
    metrics = []
    for metric_config in metrics_config:
        metric_name = metric_config["name"]
        metric_params = metric_config["params"]
        if metric_name == "pr_auc":
            metrics.append(
                PrecisionRecallAucMetric(
                    target_no_data_value=target_no_data_value, **metric_params
                )
            )
        elif metric_name.endswith("loss"):
            metrics.append(
                LossMetric(
                    target_no_data_value=target_no_data_value,
                    loss=create_loss(device, metric_name, **metric_params),
                    name=metric_name,
                )
            )
        else:
            raise ValueError(f"Unknown metric name: '{metric_name}'")

    return metrics
