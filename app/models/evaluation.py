"""Model evaluation metrics - MAE, Spearman, AUC, and Brier score."""

from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss
from scipy.stats import spearmanr


# mean absolute error
def mae(y, y_pred):
    return mean_absolute_error(y, y_pred)


# spearman rank correlation
def spearman(y, y_pred):
    return spearmanr(y, y_pred).statistic


# area under the ROC curve - measures ranking ability for classifiers
def auc(y, y_pred):
    return roc_auc_score(y, y_pred)


# brier score - measures probability calibration for classifiers
def brier(y, y_pred):
    return brier_score_loss(y, y_pred)


METRICS = {"mae": mae, "spearman": spearman, "auc": auc, "brier": brier}


# evaluates a model on a given split, printing each metric in the config
def evaluate(model, X, y, split_name, metrics):
    y_pred = model.predict(X)
    for metric in metrics:
        score = METRICS[metric](y, y_pred)
        print(f"[{split_name}] {metric}: {score:.3f}")