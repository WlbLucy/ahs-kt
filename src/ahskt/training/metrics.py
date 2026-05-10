from sklearn import metrics
from sklearn.metrics import mean_squared_error
import numpy as np


def compute_auc(targets, predictions):
    return float(metrics.roc_auc_score(targets, predictions))


def compute_accuracy(targets, predictions):
    binary_predictions = np.where(predictions > 0.5, 1.0, 0.0)
    return float(metrics.accuracy_score(targets, binary_predictions))


def compute_rmse(targets, predictions):
    return float(mean_squared_error(targets, predictions, squared=False))


def compute_binary_entropy(targets, predictions):
    loss = targets * np.log(np.maximum(1e-10, predictions)) + (1.0 - targets) * np.log(np.maximum(1e-10, 1.0 - predictions))
    return float(np.average(loss) * -1.0)


def summarize_metrics(targets, predictions):
    return {
        "loss": compute_binary_entropy(targets, predictions),
        "auc": compute_auc(targets, predictions),
        "acc": compute_accuracy(targets, predictions),
        "rmse": compute_rmse(targets, predictions),
    }
