import sys
import sqlite3

import pandas as pd


def load_data(data_filepath):
    """
    Load data and return it as a data frame.
    """
    return pd.read_csv(data_filepath, index_col='customerID')


def clean_data(data):
    """
    Clean the data.
    """
    data['SeniorCitizen'] = data['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
    data['TotalCharges'] = data['TotalCharges'].str.replace(' ', '0').astype(float)
    data.columns = [col[0].upper() + col[1:] for col in data.columns]

    return data


def feature_tenure_bucket(t):
    """Based on the tenure value, return one of the three buckets: 0-20, 21-50, 50+"""
    if t < 21:
        return '0-20'
    elif t < 51:
        return '21-50'
    else:
        return '50+'


def feature_monthlycharges_bucket(c):
    """Based on the charge value, return one of the three buckets: 0-40, 41-60, 60+"""
    if c < 41:
        return '0-40'
    elif c < 61:
        return '41-60'
    else:
        return '60+'


def feature_multiplelines_bucket(p):
    """Group customers who use only one phone line with customers without phone service."""
    if p == 'Yes':
        return 'Multiple Lines'
    else:
        return 'Other'


def feature_numinternetservices(data):
    """Create a feature that indicates a total number of internet services"""
    cols_internet_services = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    count_yes = lambda x: x.map({'No': 0, 'Yes': 1}).sum()

    return data[cols_internet_services].apply(count_yes, axis=1).astype(int)


def transform_data(data):
    """
    Transform data and create the features for the model.
    """
    data['TenureBuckets'] = data['Tenure'].apply(feature_tenure_bucket)
    data['MonthlyChargesBuckets'] = data['MonthlyCharges'].apply(feature_monthlycharges_bucket)
    data['MultipleLinesBuckets'] = data['MultipleLines'].apply(feature_multiplelines_bucket)
    data['NumInternetlServices'] = feature_numinternetservices(data)

    return data


def get_features_for_model(data):
    """
    Return a data frame with only the features that will be used for modelling.
    """
    cols_for_model = [
        'Gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'TenureBuckets', 'MonthlyChargesBuckets', 'MultipleLinesBuckets',
        'NumInternetlServices', 'Churn'
    ]
    return data[cols_for_model]


def save_data(data, database_filepath):
    """
    Save the data frame to the database
    """
    conn = sqlite3.connect(database_filepath)
    data.to_sql('data', conn, index=True, if_exists='replace')


def main():
    if len(sys.argv) == 3:
        data_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\nDATASET: {data_filepath}')
        data = load_data(data_filepath)

        print('Cleaning data...')
        data = clean_data(data)

        print('Transforming data...')
        data = transform_data(data)

        print(f'Saving data...\nDATABASE: {database_filepath}')
        save_data(data, database_filepath)

        print('Done!')
    else:
        print(
            'Please provide the filepaths of the data and the database '\
            '\nExample: '\
            'python process_data.py '\
            'WA_Fn-UseC_-Telco-Customer-Churn.csvs.csv '\
            'customers.db'
        )


if __name__ == '__main__':
    main()
