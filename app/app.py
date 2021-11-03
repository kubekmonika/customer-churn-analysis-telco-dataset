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
        """Construct the model class"""
        self.model = load_model(model_filepath.replace('.pkl', ''))
        self.summary = modelling.evaluate_model(self.model, df, outcome='return')
        self.prediction_error = self._prediction_error()

    def predict(self, data):
        """Get the verdict based on a new customer data"""
        prediction = predict_model(self.model, verbose=False, data=data)
        return {'verdict': prediction.loc[0, 'Label'], 'score': prediction.loc[0, 'Score']}

    def _prediction_error(self):
        """Summary of the prediction error"""
        prediction = predict_model(self.model, verbose=False, data=df)
        summary = prediction.groupby(['Label', 'Churn']).agg(values=('Score', 'count'))
        return summary


def validate_tenure(value):
    """Validate the tenure value"""
    if value < 0:
        raise ValueError
    elif value > 72:
        raise ValueError
    else:
        return value


def validate_charges(value):
    """Validate the monthly charges value"""
    if value < 18.25:
        raise ValueError
    elif value > 118.75:
        raise ValueError
    else:
        return value


def form_to_input_data(form):
    """Create a data frame that will be an input data for the model"""
    phone_service_flag = 'PhoneServiceEnabled' in form.keys()
    internet_service_flag = 'InternetServiceEnabled' in form.keys()

    tenure_value = validate_tenure(int(form['Tenure']))
    monthly_charges_value = validate_charges(float(form['MonthlyCharges']))

    input_data_dict = {
        'Gender': form['Gender'],
        'SeniorCitizen': form['SeniorCitizen'],
        'Partner': form['Partner'],
        'Dependents': form['Dependents'],
        'Tenure': tenure_value,
        'PhoneService': 'Yes' if phone_service_flag else 'No',
        'MultipleLines': form['MultipleLines'] if phone_service_flag else 'No phone service',
        'InternetService': form['InternetService'] if internet_service_flag else 'No',
        'OnlineSecurity': form['OnlineSecurity'] if internet_service_flag else 'No internet service',
        'OnlineBackup': form['OnlineBackup'] if internet_service_flag else 'No internet service',
        'DeviceProtection': form['DeviceProtection'] if internet_service_flag else 'No internet service',
        'TechSupport': form['TechSupport'] if internet_service_flag else 'No internet service',
        'StreamingTV': form['StreamingTV'] if internet_service_flag else 'No internet service',
        'StreamingMovies': form['StreamingMovies'] if internet_service_flag else 'No internet service',
        'Contract': form['Contract'],
        'PaperlessBilling': form['PaperlessBilling'],
        'PaymentMethod': form['PaymentMethod'],
        'MonthlyCharges': monthly_charges_value,
        'Churn': None,
    }
    input_data_df = pd.Series(input_data_dict).to_frame().T

    return input_data_df


@app.route('/')
@app.route('/index')
def index():
    """Render the home page"""
    return render_template('home.html')


@app.route('/dataset_details')
def dataset_details():
    """Render the dataset details page"""
    return render_template('dataset_details.html')


@app.route('/model_details')
def model_details():
    """Render the model details page"""
    predicted_yes = model.prediction_error.loc['Yes']
    predicted_no = model.prediction_error.loc['No']

    plot1 = gobj.Bar(
        name='Predicted: Yes',
        x=['Yes', 'No'],
        y=[predicted_yes.loc['Yes', 'values'], predicted_yes.loc['No', 'values']],
    )
    plot2 = gobj.Bar(
        name='Predicted: No',
        x=['Yes', 'No'],
        y=[predicted_no.loc['Yes', 'values'], predicted_no.loc['No', 'values']],
    )
    layout = {
        'title': 'Class prediction error',
        'xaxis': {'title': "Churn"},
        'yaxis': {'title': "Number of predicted customers", 'automargin': True},
        'barmode': 'stack',
        'width': 600,
        'height': 500,
    }
    graphs = [dict(data=[plot1, plot2], layout=layout)]

    id = "errorplot"
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        'model_details.html',
        summary=model.summary,
        id=id,
        graphJSON=graphJSON,
    )


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    """Render the prediction page"""
    if request.method == 'GET':
        return render_template(
            'prediction.html',
            basic_f=BASIC_FEATURES_UNIQUE_VALUES,
            phone_f=PHONE_FEATURES_UNIQUE_VALUES,
            internet_f=INTERNET_FEATURES_UNIQUE_VALUES,
            numerical_f=NUMERICAL_FEATURES_STATS,
        )
    elif request.method == 'POST':
        try:
            input_data = form_to_input_data(request.form)
            input_data_tuple = [(key, val[0]) for key, val in input_data.items()]
            input_data = data_processing.transform_data(input_data)
            prediction = model.predict(input_data)
            return render_template(
                'prediction.html',
                basic_f=BASIC_FEATURES_UNIQUE_VALUES,
                phone_f=PHONE_FEATURES_UNIQUE_VALUES,
                internet_f=INTERNET_FEATURES_UNIQUE_VALUES,
                numerical_f=NUMERICAL_FEATURES_STATS,
                prediction=prediction,
                data_summary=input_data_tuple,
            )
        except ValueError:
            return render_template(
                'prediction.html',
                basic_f=BASIC_FEATURES_UNIQUE_VALUES,
                phone_f=PHONE_FEATURES_UNIQUE_VALUES,
                internet_f=INTERNET_FEATURES_UNIQUE_VALUES,
                numerical_f=NUMERICAL_FEATURES_STATS,
                validation="Please provide correct values for Tenure and MonthlyCharges",
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
