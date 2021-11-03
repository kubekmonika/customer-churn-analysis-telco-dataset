import sys
import json
import plotly
import pandas as pd

sys.path.append( '..' )
import data_processing.srs as data_processing
import modelling.srs as modelling

from flask import Flask
from flask import render_template, request, jsonify, request, url_for
import plotly.graph_objs as gobj

from functools import cached_property
from pycaret.classification import *


app = Flask(__name__)


BASIC_FEATURES_UNIQUE_VALUES = {
    'Gender': ('Male', 'Female'),
    'SeniorCitizen': ('No', 'Yes'),
    'Partner': ('No', 'Yes'),
    'Dependents': ('No', 'Yes'),
    'Contract': ('Month-to-month', 'Two year', 'One year'),
    'PaperlessBilling': ('No', 'Yes'),
    'PaymentMethod': (
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)',
        'Credit card (automatic)'
    ),
}
PHONE_FEATURES_UNIQUE_VALUES = {
    'PhoneService': ('No', 'Yes'),
    'MultipleLines': ('No', 'Yes'),
}
INTERNET_FEATURES_UNIQUE_VALUES = {
    'InternetService': ('Fiber optic', 'DSL'),
    'OnlineSecurity': ('No', 'Yes'),
    'OnlineBackup': ('No', 'Yes'),
    'DeviceProtection': ('No', 'Yes'),
    'TechSupport': ('No', 'Yes'),
    'StreamingTV': ('No', 'Yes'),
    'StreamingMovies': ('No', 'Yes'),
}
NUMERICAL_FEATURES_STATS = {
    'Tenure': {'min': 0, 'max': 72},
    'MonthlyCharges': {'min': 18.25, 'max': 118.75},
}


class ModelClass:
    def __init__(self, model_filepath):
        self.model = load_model(model_filepath.replace('.pkl', ''))
        self.summary = modelling.evaluate_model(self.model, df, outcome='return')


@app.route('/')
@app.route('/index')
def index():
    return render_template('home.html')


@app.route('/dataset_details')
def dataset_details():
    return render_template('dataset_details.html')


@app.route('/model_details')
def model_details():
    # summary = modelling.evaluate_model(model, df, outcome='return')

    return render_template('model_details.html', summary=model.summary)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'GET':
        return render_template(
            'prediction.html',
            basic_f=BASIC_FEATURES_UNIQUE_VALUES,
            phone_f=PHONE_FEATURES_UNIQUE_VALUES,
            internet_f=INTERNET_FEATURES_UNIQUE_VALUES,
            numerical_f=NUMERICAL_FEATURES_STATS,
        )
    elif request.method == 'POST':
        customer_details = request.form
        return render_template(
            'prediction.html',
            basic_f=BASIC_FEATURES_UNIQUE_VALUES,
            phone_f=PHONE_FEATURES_UNIQUE_VALUES,
            internet_f=INTERNET_FEATURES_UNIQUE_VALUES,
            numerical_f=NUMERICAL_FEATURES_STATS,
            prediction=customer_details,
        )
    else:
        raise Exception('Wrong method')


def main():
    database_filepath, model_filepath = sys.argv[1:]

    global df
    global model

    # load data
    df = modelling.load_data(database_filepath)

    # load model
    model = ModelClass(model_filepath)

    # run the app
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
