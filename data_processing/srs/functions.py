import pandas as pd
import numpy as np

import seaborn as sns

import warnings
import sqlite3

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set()


def plot_distribution(data, columns_type):
    """
    Plot distribution of the data.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to plot the distribution for.
    columns_type: {'objects', 'numerical'}
        Choose which columns type to plot.

    Returns
    -------
    seaborn.axisgrid.FacetGrid
        Returns the grid of charts.
    """
    assert columns_type in {'objects', 'numerical'}, f"Wrong columns_type value: {columns_type}"

    if columns_type=='numerical':

        df = data.select_dtypes(exclude=['object']).melt()

        g = sns.FacetGrid(df, col='variable', height=4, aspect=1, col_wrap=3, sharex=False, sharey=False)
        g.map_dataframe(sns.histplot, x='value')

    elif columns_type=='objects':

        df = data.select_dtypes(include=['object']).melt()
        df['Count'] = 1

        g = sns.FacetGrid(df, col='variable', height=4, aspect=1, col_wrap=3, sharex=False, sharey=False)
        g.map_dataframe(sns.barplot, x='value', y='Count', estimator=np.sum)

        # Rotate long labels so they do not overlap
        for val, ax in g.axes_dict.items():
            if val=='PaymentMethod':
                for label in ax.get_xticklabels():
                    label.set_rotation(30)
                    label.set_ha('right')

    return g


def feature_tenure_bucket(t):
    """
    Based on the tenure value, return one of the three buckets:
    0-20, 21-50, 50+.

    Parameters
    ----------
    t : int
        A value of the tenure.

    Returns
    -------
    string
        A bucket label.
    """
    if t < 21:
        return '0-20'
    elif t < 51:
        return '21-50'
    else:
        return '50+'


def feature_monthlycharges_bucket(c):
    """
    Based on the Monthly Charges value, return one of the three buckets:
    0-40, 41-60, 60+.

    Parameters
    ----------
    c : int
        A value of the Monthly Charges.

    Returns
    -------
    string
        A bucket label.
    """
    if c < 41:
        return '0-40'
    elif c < 61:
        return '41-60'
    else:
        return '60+'


def feature_multiplelines_bucket(p):
    """
    Group customers who use only one phone line ('Yes') with customers
    without phone service ('Other').

    Parameters
    ----------
    p : string
        A value of the Multiple Lines.

    Returns
    -------
    string
        A bucket label.
    """
    if p == 'Yes':
        return 'Multiple Lines'
    else:
        return 'Other'


def feature_numinternetservices(data):
    """
    Create a feature that indicates a total number of internet services.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset.

    Returns
    -------
    pd.Series
        Number of internet services for each customer.
    """
    cols_internet_services = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    count_yes = lambda x: x.map({'No': 0, 'Yes': 1}).sum()

    return data[cols_internet_services].apply(count_yes, axis=1).astype(int)


def heatmap_churned_customers_share(data, columns):
    """
    Calculate a share of churned customers for either each value of a provided
    column or combination of values of a provided list of columns.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to plot the summary for.
    columns: string, list
        A column name or a list of column names.

    Returns
    -------
    pd.DataFrame
        A table with results.
    """
    cm = sns.light_palette("red", 100, as_cmap=True)

    _ = data.pivot_table(index=columns, columns='Churn', aggfunc='count', values='Tenure')
    summary = (_['Yes'] / _.sum(axis=1)).rename('Share of churned customers').to_frame()
    styled_summary = summary.style.background_gradient(cmap=cm, vmin=0, vmax=1)

    return styled_summary


def load_data(data_filepath):
    """
    Load data and return it as a data frame.

    Parameters
    ----------
    data_filepath : string
        Path to the database with data.

    Returns
    -------
    pd.DataFrame
        Dataset.
    """
    return pd.read_csv(data_filepath, index_col='customerID')


def clean_data(data):
    """
    Clean the data.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    data['SeniorCitizen'] = data['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
    data['TotalCharges'] = data['TotalCharges'].str.replace(' ', '0').astype(float)
    data.columns = [col[0].upper() + col[1:] for col in data.columns]

    return data


def transform_data(data):
    """
    Transform data and create the features for the model. Return a data frame
    with only the features that will be used for modelling.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset.

    Returns
    -------
    pd.DataFrame
        Transformed dataset.
    """
    cols_for_model = [
        'Gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'TenureBuckets', 'MonthlyChargesBuckets', 'MultipleLinesBuckets',
        'NumInternetlServices', 'Churn'
    ]
    data['TenureBuckets'] = data['Tenure'].apply(feature_tenure_bucket)
    data['MonthlyChargesBuckets'] = data['MonthlyCharges'].apply(feature_monthlycharges_bucket)
    data['MultipleLinesBuckets'] = data['MultipleLines'].apply(feature_multiplelines_bucket)
    data['NumInternetlServices'] = feature_numinternetservices(data)

    return data[cols_for_model]


def save_data(data, database_filepath):
    """
    Save the data to the database.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset.
    database_filepath : string
        Path where to save the database.
    """
    conn = sqlite3.connect(database_filepath)
    data.to_sql('data', conn, index=True, if_exists='replace')
