"""Model evaluation metrics - MAE and Spearman rank correlation."""

from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr


# computes and prints MAE and Spearman rank correlation for a given split
def evaluate(model, X, y, split_name):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    spearman = spearmanr(y, y_pred).statistic
    print(f"[{split_name}] MAE: {mae:.2f} | Spearman: {spearman:.3f}")