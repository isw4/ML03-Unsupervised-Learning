import pandas as pd
import numpy as np
from time import clock

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


def one_hot(num_features, train_clusters, test_clusters):
	assert train_clusters.ndim == 1 and test_clusters.ndim == 1
	enc = OneHotEncoder(categories=[np.arange(0, num_features)])
	train_clusters = np.expand_dims(train_clusters, axis=1)
	test_clusters = np.expand_dims(test_clusters, axis=1)
	train_enc = enc.fit_transform(train_clusters)
	test_enc = enc.fit_transform(test_clusters)
	return train_enc, test_enc


def fit_transform_pca(num_components, X_train, X_test):
	pca = PCA(n_components=num_components).fit(X_train)
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	return X_train_pca, X_test_pca


def fit_transform_ica(num_components, X_train, X_test):
	ica = FastICA(n_components=num_components).fit(X_train)
	X_train_ica = ica.transform(X_train)
	X_test_ica = ica.transform(X_test)
	return X_train_ica, X_test_ica


def fit_transform_rca(num_components, X_train, X_test):
	rca = GaussianRandomProjection(n_components=num_components).fit(X_train)
	X_train_rca = rca.transform(X_train)
	X_test_rca = rca.transform(X_test)
	return X_train_rca, X_test_rca


def fit_transform_rf(num_components, X_train, Y_train, X_test):
	rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
	rf.fit(X_train, Y_train)
	importances = rf.feature_importances_
	sorted_impt = np.argsort(importances)[::-1]  # Sorted in dec order
	X_train_rf = X_train.iloc[:, sorted_impt[:num_components]]
	X_test_rf = X_test.iloc[:, sorted_impt[:num_components]]
	return X_train_rf, X_test_rf


def train_predict(hidden_layer_sizes, X_train, Y_train, X_test, Y_test):
	model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=2000)
	model.fit(X_train, Y_train)
	Y_train_pred = model.predict(X_train)
	Y_test_pred = model.predict(X_test)

	train_acc = accuracy_score(Y_train, Y_train_pred)
	test_acc = accuracy_score(Y_test, Y_test_pred)

	return train_acc, test_acc


def kmeans_train_predict(k, hidden_layer_sizes, X_train, Y_train, X_test, Y_test):
	# Clustering
	model = KMeans(k, init='random', n_jobs=-1).fit(X_train)
	train_clusters = model.predict(X_train)
	test_clusters = model.predict(X_test)

	# Encoding categorical values
	train_enc, test_enc = one_hot(k, train_clusters, test_clusters)

	# Train and compute accuracy
	return train_predict(hidden_layer_sizes, train_enc, Y_train, test_enc, Y_test)


def gmm_train_predict(k, hidden_layer_sizes, X_train, Y_train, X_test, Y_test):
	# Clustering
	model = GaussianMixture(k, n_init=10).fit(X_train)
	train_clusters = model.predict(X_train)
	test_clusters = model.predict(X_test)

	# Encoding categorical values
	train_enc, test_enc = one_hot(k, train_clusters, test_clusters)

	# Train and compute accuracy
	return train_predict(hidden_layer_sizes, train_enc, Y_train, test_enc, Y_test)


def nn(X, Y):
	"""
	Runs the experiment on the old neural network architecture for the wine dataset
	1) Reduction
		- Dimension reduction on the training data
		- Transform training and testing X
		- Iteratively train NN by adding most important features one by one (PCA and RF),
		  or by transforming with increasing number of features (ICA and RP)
	2) Reduction then clustering
		- Use the same transformation as above, but pick a cluster number k
	:param X: dataframe, original X features
	:param Y: dataframe, Y labels
	"""
	print("Running neural networks on the reduced dataset")

	# Hyper-parameters
	num_folds = 10
	num_hidden_layers = 4   # Old architecture
	num_hidden_units = 25   # Old architecture
	pca_num_components = 3
	ica_num_components = 4
	rca_num_components = 5
	rf_num_components = 5
	k_kmeans = 5
	k_kmeans_pca = 4
	k_kmeans_ica = 3
	k_kmeans_rca = 3
	k_kmeans_rf = 3
	k_gmm = 3
	k_gmm_pca = 2
	k_gmm_ica = 3
	k_gmm_rca = 4
	k_gmm_rf = 3

	hidden_layer_sizes = tuple(num_hidden_units for _ in range(num_hidden_layers))

	# Opening log file
	log_path = '../logs/wine_nn.csv'
	with open(log_path, 'w') as f:
		f.write('time,train_accuracy,test_accuracy,' +
		        'pca_time,pca_train_accuracy,pca_test_accuracy,' +
		        'ica_time,ica_train_accuracy,ica_test_accuracy,' +
		        'rca_time,rca_train_accuracy,rca_test_accuracy,' +
		        'rf_time,rf_train_accuracy,rf_test_accuracy,'
		        'km_time,km_train_accuracy,km_test_accuracy,' +
		        'pca_km_time,pca_km_train_accuracy,pca_km_test_accuracy,' +
		        'ica_km_time,ica_km_train_accuracy,ica_km_test_accuracy,' +
		        'rca_km_time,rca_km_train_accuracy,rca_km_test_accuracy,' +
		        'rf_km_time,rf_km_train_accuracy,rf_km_test_accuracy,'+
		        'gmm_time,gmm_train_accuracy,gmm_test_accuracy,' +
		        'pca_gmm_time,pca_gmm_train_accuracy,pca_gmm_test_accuracy,' +
		        'ica_gmm_time,ica_gmm_train_accuracy,ica_gmm_test_accuracy,' +
		        'rca_gmm_time,rca_gmm_train_accuracy,rca_gmm_test_accuracy,' +
		        'rf_gmm_time,rf_gmm_train_accuracy,rf_gmm_test_accuracy,'+
		        '\n')

	# Training over folds
	kf = KFold(n_splits=num_folds)
	trans_time = np.zeros((num_folds, 5))
	trans_train_acc = np.zeros((num_folds, 5))
	trans_test_acc = np.zeros((num_folds, 5))
	km_time = np.zeros((num_folds, 5))
	km_train_acc = np.zeros((num_folds, 5))
	km_test_acc = np.zeros((num_folds, 5))
	gmm_time = np.zeros((num_folds, 5))
	gmm_train_acc = np.zeros((num_folds, 5))
	gmm_test_acc = np.zeros((num_folds, 5))
	fold = 0
	for train_index, test_index in kf.split(X):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

		# Dimension Reduction and transforming data to fit
		dr_time = np.zeros(5)
		pca_start = clock()
		X_train_pca, X_test_pca = fit_transform_pca(pca_num_components, X_train, X_test)
		dr_time[1] = clock() - pca_start

		ica_start = clock()
		X_train_ica, X_test_ica = fit_transform_ica(ica_num_components, X_train, X_test)
		dr_time[2] = clock() - ica_start

		rca_start = clock()
		X_train_rca, X_test_rca = fit_transform_rca(rca_num_components, X_train, X_test)
		dr_time[3] = clock() - rca_start

		rf_start = clock()
		X_train_rf, X_test_rf = fit_transform_rf(rf_num_components, X_train, Y_train, X_test)
		dr_time[4] = clock() - rf_start
		print("dim red time: {}".format(dr_time))

		# Fitting and computing accuracy (after dimension reduction only)
		transformed_train = [X_train, X_train_pca, X_train_ica, X_train_rca, X_train_rf]
		transformed_test = [X_test, X_test_pca, X_test_ica, X_test_rca, X_test_rf]
		for i in range(0, 5):
			TX_train = transformed_train[i]
			TX_test = transformed_test[i]
			start_time = clock()
			trans_train_acc[fold, i], trans_test_acc[fold, i] = train_predict(hidden_layer_sizes,
			                                                                  TX_train, Y_train, TX_test, Y_test)
			time = clock() - start_time
			print("{} + {}".format(dr_time[i], time))
			trans_time[fold, i] = dr_time[i] + time
		print("dim red + training time: {}".format(trans_time[fold, :]))

		# Clustering the reduced dataset and computing accuracy (dim red -> clustering)
		k_for_kmeans = [k_kmeans, k_kmeans_pca, k_kmeans_ica, k_kmeans_rca, k_kmeans_rf]
		k_for_gmm = [k_gmm, k_gmm_pca, k_gmm_ica, k_gmm_rca, k_gmm_rf]
		for i in range(0, 5):
			TX_train = transformed_train[i]
			TX_test = transformed_test[i]
			start_time = clock()
			km_train_acc[fold, i], km_test_acc[fold, i] = kmeans_train_predict(k_for_kmeans[i], hidden_layer_sizes,
			                                                                   TX_train, Y_train, TX_test, Y_test)
			time = clock() - start_time
			print("{} + {}".format(dr_time[i], time))
			km_time[fold, i] = dr_time[i] + time

			start_time = clock()
			gmm_train_acc[fold, i], gmm_test_acc[fold, i] = gmm_train_predict(k_for_gmm[i], hidden_layer_sizes,
			                                                                   TX_train, Y_train, TX_test, Y_test)
			gmm_time[fold, i] = dr_time[i] + (clock() - start_time)
		print("dim red + km + training time: {}".format(km_time[fold, :]))
		print("dim red + gmm + training time: {}".format(gmm_time[fold, :]))

		fold += 1

	# Finding mean over all trials and then output
	trans_time = trans_time.mean(axis=0)
	trans_train_acc = trans_train_acc.mean(axis=0)
	trans_test_acc = trans_test_acc.mean(axis=0)
	km_time = km_time.mean(axis=0)
	km_train_acc = km_train_acc.mean(axis=0)
	km_test_acc = km_test_acc.mean(axis=0)
	gmm_time = gmm_time.mean(axis=0)
	gmm_train_acc = gmm_train_acc.mean(axis=0)
	gmm_test_acc = gmm_test_acc.mean(axis=0)

	with open(log_path, 'a') as f:
		output = ''
		for i in range(0, 5):
			output = output + ',{},{},{}'.format(trans_time[i], trans_train_acc[i], trans_test_acc[i])
		for i in range(0, 5):
			output = output + ',{},{},{}'.format(km_time[i], km_train_acc[i], km_test_acc[i])
		for i in range(0, 5):
			output = output + ',{},{},{}'.format(gmm_time[i], gmm_train_acc[i], gmm_test_acc[i])
		output = output[1:] + '\n'  # Removing the first ',' and adding a new line
		f.write(output)


def main():
	processed_data_path = '../data/processed-wine-equality-red.csv'
	df = pd.read_csv(processed_data_path)
	X = df.iloc[:, :-1]
	Y = df.iloc[:, -1]
	nn(X, Y)

if __name__ == "__main__":
	main()