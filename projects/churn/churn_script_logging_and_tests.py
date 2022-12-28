"""
Script for testing functions in churn_library.py and performing loggin.

Author: Nasser Boan
Date: December, 2022
"""

# import logging
import logging
logging.basicConfig(
    filename='logs/my_log.log',
    level = logging.INFO,
    filemode='w+',
    format='%(name)s - %(levelname)s - %(message)s')

# importing libraries
import os
import pandas as pd
import churn_library as cls
from pytest import fixture


@fixture
def data_path():
	return 'data/bank_data.csv'

@fixture
def data_not_encoded(data_path):
	return cls.import_data(data_path)

@fixture
def eda_file_names():
	return ['churn_distribution.png','customer_age_distribution.png',
	'marital_status_distribution.png','total_transaction_distribution.png',
	'heatmap.png']

@fixture
def columns_to_encode():
	return ['Gender', 'Education_Level', 
	'Marital_Status', 'Income_Category', 
	'Card_Category']

@fixture
def response_variable_name():
	return 'churn'

@fixture
def model_features():
	return [
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

def test_import(data_path):
	'''
	test data import - this example is completed for you to assist with the 
	other test functions
	'''
	try:
		df = cls.import_data(data_path)
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(eda_file_names):
	'''
	test perform eda function
	'''
	
	for file in eda_file_names:
		try:
			assert os.path.exists('images/eda/'+file)
		except:
			raise FileNotFoundError('The {} was not found'.format(file))

	

def test_encoder_helper(data_not_encoded, columns_to_encode, response_variable_name):
	'''
	test encoder helper
	'''

	data_encoded = cls.encoder_helper(
		data_not_encoded,
		columns_to_encode,
		response_variable_name
		)

	for encoded_column in columns_to_encode:
		try:
			assert encoded_column+'_churn' in data_encoded
		except:
			raise KeyError('The {}_churn was not found.'.format(encoded_column))

def test_perform_feature_engineering(data_not_encoded, response_variable_name, model_features):
	'''
	test perform_feature_engineering
	'''

	model_training_data = cls.perform_feature_engineering(data_not_encoded,response_variable_name)


	## testing if the function returns all four datasets
	try:
		assert len(model_training_data) == 4
	except:
		raise FileNotFoundError('The function must return all four datasets.')


	## testing if X_train, X_test are the correct type
	for dataset in (model_training_data[0],model_training_data[1]):
		try:
			assert isinstance(dataset,pd.core.frame.DataFrame)
		except:
			raise TypeError('X_train and X_test must be instances of a pandas Dataframe.')

	## testing if y_train, y_test are the correct type
	for dataset in (model_training_data[2],model_training_data[3]):
		try:
			assert isinstance(dataset,pd.core.series.Series)
		except:
			raise TypeError('y_train and y_test must be instances of a pandas Series.')


	## testing if X_train and X_test have all the necessary columns
	for feature in model_features:
		try:
			assert feature in model_training_data[0]
		except:
			raise KeyError('The {} was not found in X_train.'.format(feature)) 
		
		try:
			assert feature in model_training_data[1]
		except:
			raise KeyError('The {} was not found in X_test.'.format(feature)) 

def test_train_models():
	'''
	test train_models
	'''



	pass

if __name__ == "__main__":
	pass








