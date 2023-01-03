# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project implements a library that: Imports, treats and create the necessary features for a training, evaluating and saving 2 ML Models that can accuratly predict if a costumer will churn or not.

## Files and data description

```
.
├── data                 # Data used in the process of training the models
│   └── bank_data.csv
├── images               # Store EDA and model evaluation results charts
│   ├── eda
│   └── results
├── logs                 # Store logs
│   └── churn_library.log
├── models               # Store models
│   ├── logistic_regression.pkl
│   └── random_forest_model.pkl
├── models               # Notebooks used for creating the library
│   └── churn_notebook.ipynb
├── churn_library.py     # Library for the churn solution.
├── churn_script_logging_and_tests.py # File that implements the testing and logging of churn_library functions.
├── README.md            # The file that you're reading right now.
└── requirements_py3.6.txt # Environment file (using python 3.6)
```

## Running Files

To run this project first you should create and activate an environment.

Using <code>conda</code>:

```conda create -n <ENV_NAME> python=3.6```

```conda activate <ENV_NAME>```

Use <code>pip</code> to install the necessary requirements.

```pip install -r requirements_py3.6.txt```

Now you should be able to run the whole project

Use:

```ipython churn_library.py```

To run the library and

```ipython churn_script_logging_and_tests.py```

To run the tests and log and the information about the tests.