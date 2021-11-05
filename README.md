# Predict churn of customers

*This is a capstone project made for the Udacity's Data Scientist Nanodegree program.*

## Writeup

### Project overview

Churn is the phenomenon where customers stop using your product. It is widely known in business that it it easier to keep an existing customer than to gain a new one. Similarly, it is also easier to save a customer before they leave than to convince them to come back. Hence, understanding and preventing customer churn is a crucial task and every business should allocate some part of their resources to work on it. Customer churn analysis helps in finding indicators of churn. Using such insights you can improve your service and offers which will lead to better user retention.

In this project I will take on a role of a data scientist that works for Telco, which is a fictional telecommunications company that provides home phone and Internet services. I am provided with a dataset of current and former customers, that contains information about:

* Customers who left within the last month – the column is called Churn
* Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
* Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
* Demographic info about customers – gender, age range, and if they have partners and dependents

Source of the data: [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)

### Problem statement

In order to help the company in retaining the customers I need to:

1. Build a binary classification model that will predict which customers may churn. - Such model can be used for example to predict what will happen in the near future. Knowing what churn we expect to see will help in building business strategy.
2. Build a web application that will allow to make predictions based on entered data. - One data scientist is a bottleneck when there are many stakeholders who need data. Such an application will be helpful for example for the team who works closely with individual customers. When they know the probability of churn of a given customer, they may act accordingly and customize the offer.

### Evaluation metrics

To evaluate the model we should choose metrics that are relevant to the problem and the dataset.

*Accuracy* is not a good metric to use in our case, because the target feature (`churn`) is not balanced throughout the dataset - 26.5% of customers are labeled as churned and 73.5% are not. This means that if we label all the customers as not churning, then we will have 73% accuracy - even though this number looks impressive, the model is useless.

In our problem we want to detected as many customers who may churn as possible, so that we can act and prevent it (that is *precision*). On the other hand, we do not want to bother too many customers who do not plan to leave our service (that is *recall*). Hence, we need to find a model with good balance between *precision* and *recall*, and also optimise the model for these metrics. The *F-score* is a way of combining the precision and recall of the model, and it is defined as the harmonic mean of the model’s precision and recall. Later, the *F-score* will be used to compare models and choosing one.

### Analysis

The exploratory data analysis (EDA) is summarised in the `eda.ipynb` notebook which can be found in the `data_processing` directory.

### Data preprocessing

Based on the EDA I made a decision to transform some of the features.

I decided to skip the `Total Charges` feature as it is highly correlated with `Tenure` (correlation 0.73), and when we combine `Tenure` with `Monthly Charges` we basically get the same information.

The numeric characteristics of the customers can be divided into ranges related to lower or higher churn. Hence, I decided to replace them with buckets as follows:
* `Tenure` values are divided into following ranges:
  * 0-20: related to high churn
  * 21-50: related to medium churn
  * 50+: related to low churn
* `Monthly Charges` values are divided into following ranges:
  * 0-40: with low churn
  * 41-60: with medium churn
  * 60+: with high churn

For the `Multiple Lines` feature I grouped two categories, `No multiple lines` and `No phone service`, into one `Other`, as they did not have a significant difference in their relation to churn.

And finally, I created a new feature that indicates a total number of internet services a customer has.

In the end, the following set of features was chosen to build the model:
* `Gender`: The customer’s gender: Male, Female
* `Senior Citizen`: Indicates if the customer is 65 or older: Yes, No
* `Partner`: Indicates if the customer is a partner: Yes, No
* `Dependents`: Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc.
* `Phone Service`: Indicates if the customer subscribes to home phone service with the company: Yes, No
* `Internet Service`: Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.
* `Online Security`: Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No
* `Online Backup`: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No
* `Device Protection Plan`: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No
* `Tech Support`: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No
* `Streaming TV`: Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service
* `Streaming Movies`: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service
* `Contract`: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year
* `Paperless Billing`: Indicates if the customer has chosen paperless billing: Yes, No
* `Payment Method`: Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check
* `Tenure Buckets`: Indicates the range in which the customer's tenure value is, it is denoted in months: 0-20, 21-50, 50+
* `Monthly Charges Buckets`: Indicates a range in which the customer’s current total monthly charge is for all their services from the company: 0-40, 41-60, 60+
* `Multiple Lines Buckets`: Indicates if the customer subscribes to multiple telephone lines with the company: Yes; either has one line or not at all: Other
* `Num Internet Services`: Indicates the total number of additional internet services the customer has: 0 - 6
* `Churn`: Indicates if the customer have churned: Yes, No

### Implementation and refinement

The dataset is split into three parts:
* train - to train the models,
* test - to test the trained models,
* validation - to perform the final validation of the model's performance on previously unseen data.

In training all the models, the 5-fold cross-validation is used. We do not need to perform any imputation method, as no missing values occur in this dataset.

To quickly iterate through various models I used the [PyCaret](https://pycaret.org/) library. It is an open-source, low-code machine learning library in Python that automates machine learning workflows.

With basic setup the results looked as follows:

<img src="modelling/images/model_iter_01.png" alt="Models comparison - basic setup" width="600"/>

After testing some tweaks, I found that the best results gives the following configuration:
* the feature `NumInternetServices` is chosen to be numerical, the rest is categorical,
* the numeric feature is normalised, the `z-score` normalisation is used,
* SMOTE method is used to fix the imbalance.

And here are the results:

<img src="modelling/images/model_iter_02.png" alt="Models comparison - final setup" width="600"/>

We see that the later setup improved the F1 score for the top three models.

After settling on the features, I moved to the next step which is tuning the hyper-parameters.
Firstly, I searched for best parameters. Secondly, I tested the models on test data.

Here are the statistics of the tuned models.

**Logistic regression**

```md
+---------------+------------+----------+-------------+-------+
|               |   Accuracy |   Recall |   Precision |    F1 |
+===============+============+==========+=============+=======+
| Model Summary |      0.765 |    0.825 |       0.557 | 0.665 |
+---------------+------------+----------+-------------+-------+
```

**Ridge classifier**

```md
+---------------+------------+----------+-------------+-------+
|               |   Accuracy |   Recall |   Precision |    F1 |
+===============+============+==========+=============+=======+
| Model Summary |      0.763 |    0.833 |       0.554 | 0.665 |
+---------------+------------+----------+-------------+-------+

```

**Linear SVM**

```md
+---------------+------------+----------+-------------+-------+
|               |   Accuracy |   Recall |   Precision |    F1 |
+===============+============+==========+=============+=======+
| Model Summary |      0.694 |    0.914 |       0.479 | 0.628 |
+---------------+------------+----------+-------------+-------+
```

**KNN classifier**

```md
+---------------+------------+----------+-------------+-------+
|               |   Accuracy |   Recall |   Precision |    F1 |
+===============+============+==========+=============+=======+
| Model Summary |      0.766 |    0.774 |       0.563 | 0.652 |
+---------------+------------+----------+-------------+-------+
```

**Decision tree**

```md
+---------------+------------+----------+-------------+-------+
|               |   Accuracy |   Recall |   Precision |    F1 |
+===============+============+==========+=============+=======+
| Model Summary |      0.763 |    0.769 |        0.56 | 0.648 |
+---------------+------------+----------+-------------+-------+
```

All the models give similar results when it comes to F-score. I choose the logistic regression as the final model for the problem - it is a simple model which is easy to interpret and can be trained quickly.

Here are the statistics describing the linear regression model which was trained using the entire train+test dataset and validated on unseen data:

```md
+---------------+------------+----------+-------------+------+
|               |   Accuracy |   Recall |   Precision |   F1 |
+===============+============+==========+=============+======+
| Model Summary |      0.744 |    0.726 |       0.511 |  0.6 |
+---------------+------------+----------+-------------+------+
```

The visualisation of the class prediction error, where `1` means *churned* and `0` means *not churned*.

<img src="modelling/images/Prediction Error.png" alt="Class prediction error" width="580"/>

The visualisation of the cofusion matrix.

<img src="modelling/images/Confusion Matrix.png" alt="Class prediction error" width="500"/>

The top 10 features according to their importance.

<img src="modelling/images/Feature Importance.png" alt="Class prediction error" width="700"/>

We can conclude that churn is mostly determined by the following characteristics of
the customer:
* has short tenure (high churn),
* uses the fiber optic (high churn),
* the contract is month-to-month (high churn) or two-year (low churn).

In order to obtain better results we would have to enrich this dataset with additional data
about customers or come up with new features that would give us more relevant information.

## Web app

<img src="app/images/home.png" alt="Home Page" width="600"/>

When you open the web app, you will see a home page with a short introduction. There
will be also three navigation buttons that will redirect you to other secions.

##### Dataset details

Here you will see a description of the dataset as well as a graphical summary of it.

<img src="app/images/dataset.png" alt="Home Page" width="600"/>

##### Model summary

Here you will see basic statistics of the model.

<img src="app/images/model.png" alt="Home Page" width="600"/>

##### Make your own prediction

In this section you will find a form with customer characteristics. When you fill this form and click the `Submit data` button, you will be presented with a prediction result for the provided characteristics.

<img src="app/images/prediction_form.png" alt="Home Page" width="600"/>

<img src="app/images/prediction_result.png" alt="Home Page" width="600"/>
