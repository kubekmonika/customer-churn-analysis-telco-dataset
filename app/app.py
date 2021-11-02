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

from collections import defaultdict
from pycaret.classification import *


app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('home.html')


@app.route('/dataset_details')
def dataset_details():
    return render_template('dataset_details.html')


@app.route('/model_details')
def model_details():
    summary = modelling.evaluate_model(model, df, outcome='return')

    return render_template('model_details.html', summary=summary)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'GET':
        return render_template('prediction.html')
    elif request.method == 'POST':
        customer_details = (
            request.form['title'],
            request.form['name'],
            request.form['comment']
            )
        insert_comment(user_details)
        return render_template('prediction.html')
    else:
        raise Exception('Wrong method')


def main():
    database_filepath, model_filepath = sys.argv[1:]

    global df
    global model

    # load data
    df = modelling.load_data(database_filepath)
    print(df.columns)

    # load model
    model = load_model(model_filepath.replace('.pkl', ''))
    print(model)

    # run the app
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
