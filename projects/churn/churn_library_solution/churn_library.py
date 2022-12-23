# library doc string

"""
Library that imports, pre-process data, performs feature engineering, train and
evaluate a random forest and logistic regression model.
"""

# import libraries
import time
import os
import sys
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    Read the csv data in pth, creates the 'churn' column and returns the resulting dataframe.

    Args:
        pth: (str) a path to the csv file

    Returns:
        df: (pandas.core.frame.DataFrame) a pandas dataframe with the 'churn' column.

    '''

    churn_data = pd.read_csv(pth)
    churn_data['churn'] = churn_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return churn_data


def perform_eda(df):
    '''
    Perform exploratory data analysis on 'df' and save figures to images folder.

    Args:
        df: (pandas.core.frame.DataFrame) pandas dataframe with the 'churn' column.

    Returns:
        None
    '''

    plt.figure(figsize=(20, 10))
    df['churn'].hist()
    plt.savefig('./images/eda/churn_distribution')

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_distribution')

    plt.figure(figsize=(20, 10))
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_distribution')

    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], kde=True)
    plt.savefig('./images/eda/total_transaction_distribution')

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap')


def _find_category_churn(df, category_name, response):
    '''
    Private function for finding a category churn.

    Args:
        df: (pandas.core.frame.DataFrame) pandas dataframe with the 'churn'
        column and category information.
        category_name: (str) a string describing the dataframe column
        to be encoded.

    Returns:
        result: (dict) a dictionary with each category and propotion of
        churn in that category as key:value pair.
    '''

    churn_info = df.groupby(category_name)[response].mean()
    result = dict(zip(churn_info.index, churn_info.values))

    return result


def encoder_helper(df, category_lst, response):
    '''
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category.

    Args:
        df: (pandas.core.frame.DataFrame) pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be
        used for naming variables or index y column.

    Returns:
        df: (pandas.core.frame.DataFrame) pandas dataframe with encoded columns
    '''

    for category in category_lst:
        mapping_information = _find_category_churn(df, category, response)
        df[category + f'_churn'] = df[category].map(mapping_information)

    return df


def perform_feature_engineering(df, response):
    '''
    Performs the necessary feature engineering and returns the prepared
    datasets for model training.

    Args:
        df: (pandas.core.frame.DataFrame) pandas dataframe
        response:  (str) string of response name [optional argument that
        could be used for naming variables or index y column]

    Returns:
        X_train: (pandas.core.frame.DataFrame) X training data
        X_test: (pandas.core.frame.DataFrame) X testing data
        y_train: (pandas.core.series.Series) y training data
        y_test: (pandas.core.series.Series) y testing data
    '''

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_churn',
        'Education_Level_churn',
        'Marital_Status_churn',
        'Income_Category_churn',
        'Card_Category_churn']

    dataframe_encoded = encoder_helper(df,
                                       ['Gender',
                                        'Education_Level',
                                        'Marital_Status',
                                        'Income_Category',
                                        'Card_Category'],
                                       response)

    X = dataframe_encoded[keep_cols]
    y = dataframe_encoded[response]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    Produces classification report for training and testing results
    and stores report as image in images folder.

    Args:
        y_train: (numpy.ndarray) training response values
        y_test:  (numpy.ndarray) test response values
        y_train_preds_lr: (numpy.ndarray) training predictions
        from logistic regression
        y_train_preds_rf: (numpy.ndarray) training predictions
        from random forest
        y_test_preds_lr: (numpy.ndarray) test predictions
        from logistic regression
        y_test_preds_rf: (numpy.ndarray) test predictions
        from random forest

    Returns:
        None
    '''
    plt.figure(figsize=(12, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/rf_results')

    plt.figure(figsize=(12, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/logistic_results')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store the figure

    output:
        None
    '''

    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # saving the plot
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        grid_random_forest: (sklearn.ensemble._forest.RandomForestClassifier)
        a random forest model optimized with best parameters
        logistic_regression: (sklearn.linear_model._logistic.LogisticRegression)
        a trained logistic regression model

    '''

    ###### RANDOM FOREST ######
    # defining the random forest model
    random_forest = RandomForestClassifier(random_state=42)

    # which parameters to test
    parameters_to_test = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # finding the best estimator
    grid_random_forest = GridSearchCV(
        estimator=random_forest,
        param_grid=parameters_to_test,
        cv=5)
    grid_random_forest.fit(X_train, y_train)

    # generating the train and test predictions
    y_train_preds_random_forest = grid_random_forest.best_estimator_.predict(
        X_train)
    y_test_preds_random_forest = grid_random_forest.best_estimator_.predict(
        X_test)

    # saving the model
    joblib.dump(grid_random_forest.best_estimator_,
                './models/random_forest_model.pkl')

    ###### LOGISTIC REGRESSION ######
    # defining the model
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=3000)
    logistic_regression.fit(X_train, y_train)

    y_train_preds_logistic_regression = logistic_regression.predict(X_train)
    y_test_preds_logistic_regression = logistic_regression.predict(X_test)

    # saving the model
    joblib.dump(logistic_regression, './models/logistic_regression.pkl')

    # ROC AUC
    f, ax = plt.subplots(figsize=(15, 8))
    plot_roc_curve(grid_random_forest.best_estimator_,
                   X_test, y_test, ax=ax, alpha=0.8)
    plot_roc_curve(logistic_regression, X_test, y_test, ax=ax, alpha=0.8)
    f.savefig('./images/results/roc_curve_result')

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_logistic_regression,
        y_train_preds_random_forest,
        y_test_preds_logistic_regression,
        y_test_preds_random_forest)
    feature_importance_plot(
        grid_random_forest,
        X_train,
        './images/results/feature_importances')


if __name__ == "__main__":
    FILE_PATH = sys.argv[1]

    start = time.time()

    print(f'>> STARTING AT {start}')
    print('>> IMPORTING DATA')
    DATA = import_data(FILE_PATH)

    print('>> PERFORMING EDA')
    perform_eda(DATA)

    print('>> PERFORMING FEATURE ENGINEERING')
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        DATA, 'churn')

    print('>> TRAINING MODELS')
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)

    print('>> DONE')

    print('Duration: {} seconds'.format(time.time() - start))
