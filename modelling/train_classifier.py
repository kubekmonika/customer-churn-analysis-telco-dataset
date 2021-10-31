import sys
import sqlite3
import pandas as pd
import numpy as np
import srs

from pycaret.classification import *

import warnings
warnings.simplefilter(action='ignore')


def load_data(database_filepath):
    """
    Load the data and split it into messages and categories.
    """
    conn = sqlite3.connect(database_filepath)
    data = pd.read_sql('SELECT * FROM data', conn, index_col='customerID')

    return data


def build_model(data):
    """
    Build the ML pipeline and a model.
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
            model, n_iter=1000, optimize='f1', verbose=False,
            custom_grid=custom_grid
        )

        final_model = finalize_model(tuned_model)

    return final_model


def save_model_(model, model_filepath):
    """
    Save the model as a pickle file.
    """
    save_model(model, model_filepath, model_only=False)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        model_filepath = model_filepath.replace('.pkl', '')

        print(f'Loading data...\nDATABASE: {database_filepath}')
        data = load_data(database_filepath)

        print('Building a model...')
        model = build_model(data)
        print(model)

        print('Evaluating model...')
        srs.evaluate_model(model)

        print(f'Saving model...\nMODEL: {model_filepath}.pkl')
        save_model_(model, model_filepath)

        print('Done!')
    else:
        print(
            'Please provide the filepath of the database '\
            'and the filepath of the model'\
            '\nExample: '\
            'python train_classifier.py '\
            'customers.db '\
            'model.pkl'
          )


if __name__ == '__main__':
    main()
