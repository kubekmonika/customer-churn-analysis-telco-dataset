import pandas as pd
import numpy as np

from sklearn import metrics
from pycaret.classification import predict_model


def evaluate_model(model, data=None):
    """
    Show the summary statistics of the model.
    """
    if type(data) is pd.DataFrame:
        predictions = predict_model(model, verbose=False, data=data)
    else:
        predictions = predict_model(model, verbose=False)

    summary = {}
    y_true = predictions['Churn'].map({'Yes': 1, 'No': 0}).values
    y_pred = predictions['Label'].map({'Yes': 1, 'No': 0}).values

    summary['Accuracy'] = metrics.accuracy_score(y_true, y_pred)
    summary['Recall'] = metrics.recall_score(y_true, y_pred)
    summary['Precision'] = metrics.precision_score(y_true, y_pred)
    summary['F1'] = metrics.f1_score(y_true, y_pred)

    df = pd.DataFrame(summary, index=['Model Summary']).round(3)

    print(df.to_markdown(tablefmt="grid"))
