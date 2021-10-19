# Predict churn of customers

*This is a capstone project made for the Udacity's Data Scientist Nanodegree program.*

## Background information

### What is churn?

Churn is the phenomenon where customers stop using your product. It is often presented as a metric that quantifies how many people have stopped using your service over a specific time period. This kind of analysis is important as it helps you find indicators of lower engagement and spot the people who may leave. Using such insights you can improve your service and offers which will lead to better user retention.

### Dataset

The *Telco customer churn* data contains information about a fictional telecommunications company that provided home phone and Internet services.

Source: [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)

Each row represents a customer, each column contains customer’s attributes.

The data set includes information about:

* Customers who left within the last month – the column is called Churn
* Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
* Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
* Demographic info about customers – gender, age range, and if they have partners and dependents

### The aim of this project

1. Do exploratory data analysis of the dataset.
2. Build a machine learning model that will predict customer churn.
3. Build a web application that will allow to make predictions based on entered data.

## Project's file structure

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- WA_Fn-UseC_-Telco-Customer-Churn  # data to process

- notebooks
|- eda.ipynb  # Containts exploratory data analysis

- models
|- train_classifier.py  # script that creates the ML model
|- classifier.pkl  # saved model 

- README.md
- requirements.txt
```

## Results

TBD

## Web app

The web app displays some visualizations of the data.

TBD

## How to run it

### Installation

Libraries and their versions required for replication of this analysis are listed in the `requirements.txt` file.

Python version: 3.8.11

Run `conda create --name <env> --file requirements.txt` to create a conda environment, and then `conda activate <env>` to activate it.
