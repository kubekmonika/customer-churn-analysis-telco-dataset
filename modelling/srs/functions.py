import sqlite3
import warnings

import pandas as pd
import numpy as np

from sklearn import metrics
from pycaret.classification import *

warnings.simplefilter(action='ignore')


def evaluate_model(model, data=None):
    """
    Show the summary statistics of the model. By default the test data is used,
    but you can provide other dataset.

    Parameters
    ----------
    model: scikit-learn compatible object
        Trained model object.
    data: pd.DataFrame, default None
        Other data to use for evaluation.
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


def load_data(database_filepath):
    """
    Load the data and split it into messages and categories.

    Parameters
    ----------
    database_filepath: string
        Path to the databse with data.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    conn = sqlite3.connect(database_filepath)
    data = pd.read_sql('SELECT * FROM data', conn, index_col='customerID')

    return data


def build_model(data):
    """
    Build the ML pipeline and a model, using 5-fold cross-validation and with
    hypoerparameters tuning.

    Parameters
    ----------
    data: pd.DataFrame
        Dataset.

    Returns
    -------
    model
        Trained model object.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=UserWarning,
        )
        setup(
            data,
            target='Churn',
            n_jobs=5,
            train_size=0.8,
            silent=True,
            verbose=False,
            html=False,
            numeric_features=['NumInternetlServices'],
            normalize=True,
            fix_imbalance=True,
        )

        model = create_model('lr', fold=5, verbose=False)

        custom_grid = {
            'fit_intercept': [True],
            'solver': ['saga'],
            'penalty': ['elasticnet'],
            'C': np.logspace(0.0001, 4, num=50) / 100,
            'class_weight': ['balanced'],
            'dual': [False],
            'max_iter': [1000],
            'l1_ratio': [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
        }
        tuned_model = tune_model(
            model, n_iter=100, optimize='f1', verbose=False,
            custom_grid=custom_grid
        )

        final_model = finalize_model(tuned_model)

    return final_model


def save_model_(model, model_filepath):
    """
    Save the model as a pickle file.

    Parameters
    ----------
    model: scikit-learn compatible object
        Trained model object.
    model_filepath: string
        Path where to save the model object.
    """
    save_model(model, model_filepath, model_only=False)
