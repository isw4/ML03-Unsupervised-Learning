import pandas as pd
import numpy as np


def drop_empty_rows(df):
	row_no_nan = df.notna().all(axis=1)
	new_df = df[row_no_nan]
	return new_df


def normalize(X):
	# Mean normalization
	return (X - X.mean()) / X.std()


def wine_aggregate_labels(Y):
	new_Y = Y.copy()
	new_Y[Y >= 6] = 1  # Good wines
	new_Y[Y < 6] = 0  # Bad wines
	return new_Y


def wine_process():
	"""
	reads csv file and removes data instances with empty features, normalizes, and aggregates labels
	then outputs them into a csv file with ',' delimiter
	"""
	filepath_input = '../data/raw-winequality-red.csv'           # Raw data
	filepath_output = '../data/processed-wine-equality-red.csv'  # Completely processed data

	df = pd.read_csv(filepath_input, sep=';')

	# Removes all rows that have empty cells
	df = drop_empty_rows(df)

	df.iloc[:, :-1] = normalize(df.iloc[:, :-1])
	df.iloc[:, -1] = wine_aggregate_labels(df.iloc[:, -1])

	df.to_csv(filepath_output, index=False)


def pet_aggregate_labels(Y):
	Y_copy = Y.copy().str.strip().str.lower()
	def transform_labels(label):
		if label == 'adoption' or label == 'return_to_owner':
			return 0
		elif label == 'died' or label == 'euthanasia' or label == 'transfer':
			return 1
		return None
	Y_copy = Y_copy.apply(transform_labels)
	return Y_copy


def pet_process():
	"""
	reads csv file and encodes features, handles null values, normalizes features, and aggregates labels
	then outputs them into a csv file with ',' delimiter
	"""
	filepath_input = '../data/raw-pet-outcomes.csv'         # Raw data
	filepath_output = '../data/processed-pet-outcomes.csv'  # Completely processed data

	### Read CSV and extract relevant features and the outcome
	df = pd.read_csv(filepath_input)
	df = df.loc[:, ['Name', 'AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'OutcomeType']]

	### Encoding features and dropping rows with empty cells
	# Encoding names into whether the pet has a name (pets with no name will be NaN)
	has_name = ~df['Name'].isna()
	df['Named'] = has_name
	df = df.drop(columns='Name')

	# Dropping rows with missing features
	df = drop_empty_rows(df)

	# Encoding pet type(Dog or Cat) into whether the pet is a dog (only other type is cat)
	is_dog = df['AnimalType'].str.strip().str.lower() == 'dog'
	df['isDog'] = is_dog
	df = df.drop(columns='AnimalType')

	# Transforming age from str to datetime. The types of ages that are available are given in 'x years',
	# 'x months', 'x weeks', 'x days'. We will transform this to days, following the assumption that
	# 1 year = 365.25 days
	# 1 month = 30.42 days
	# 1 week = 7 days
	raw_age = df['AgeuponOutcome'].str.strip().str.lower().str.split(' ')
	def transform_age(pair):
		number = int(pair[0])
		time_unit = pair[1]
		if number == 0:
			return None
		elif time_unit == 'day' or time_unit == 'days':
			return int(number)
		elif time_unit == 'week' or time_unit == 'weeks':
			return int(number * 7)
		elif time_unit == 'month' or time_unit == 'months':
			return int(number * 30.42)
		elif time_unit == 'year' or time_unit == 'years':
			return int(number * 365.25)
		return None
	df['AgeuponOutcome'] = raw_age.apply(transform_age)

	# Dropping rows with age = NaN
	df = drop_empty_rows(df)

	# Processing Sex (values will be 'spayed/intact female' or 'spayed/intact male' or 'unknown')
	raw_sex = df['SexuponOutcome'].str.strip().str.lower().str.split(' ')
	def transform_sex(pair):
		if len(pair) == 2:
			if pair[1] == 'male':
				return True
			if pair[1] == 'female':
				return False
		return None
	df['MaleuponOutcome'] = raw_sex.apply(transform_sex)
	df = df.drop(columns='SexuponOutcome')

	# Dropping rows with unknown sex
	df = drop_empty_rows(df)

	# Arranging the column ordering before adding the breed one-hot encoding
	df = df[['Named', 'isDog', 'AgeuponOutcome', 'MaleuponOutcome', 'OutcomeType']]

	### Normalizing continuous features and aggregating labels
	df['AgeuponOutcome'] = normalize(df['AgeuponOutcome'])
	df['OutcomeType'] = pet_aggregate_labels(df['OutcomeType'])

	df.to_csv(filepath_output, index=False)


if __name__ == "__main__":
	wine_process()
	pet_process()