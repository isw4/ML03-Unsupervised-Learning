import pandas as pd
import numpy as np
from time import clock

from scipy.stats import kurtosis
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, silhouette_score, accuracy_score

import utils


def kmeans(X, Y):
	print("Running K-means")

	# Range of k to test
	krange = np.arange(2, 50)

	# Opening log file
	log_path = '../logs/pet_kmeans.csv'
	with open(log_path, 'w') as f:
		f.write('k,time,ari,homogeneity,completeness,silhouette,distortion\n')

	for k in krange:
		# Computing k means
		start_time = clock()
		kmeans_model = KMeans(k, init='random', n_jobs=-1).fit(X)
		clusters = kmeans_model.predict(X)
		time_taken = clock() - start_time

		# Computing metrics
		ari = adjusted_rand_score(Y, clusters)
		hom = homogeneity_score(Y, clusters)
		com = completeness_score(Y, clusters)
		sil = silhouette_score(X, clusters)    # Euclidean distance
		dis = kmeans_model.inertia_

		# Logging metrics
		with open(log_path, 'a') as f:
			out = '{},{},{},{},{},{},{}\n'.format(k, time_taken, ari, hom, com, sil, dis)
			f.write(out)


def gmm(X, Y):
	print("Running GMM")

	# Range of k to test
	krange = np.arange(2, 50)

	# Opening log file
	log_path = '../logs/pet_gmm.csv'
	with open(log_path, 'w') as f:
		f.write('k,time,ari,homogeneity,completeness,silhouette,aic,bic\n')

	for k in krange:
		# Computing GMM
		start_time = clock()
		gmm_model = GaussianMixture(k, n_init=10).fit(X)
		clusters = gmm_model.predict(X)
		time_taken = clock() - start_time

		# Computing metrics
		ari = adjusted_rand_score(Y, clusters)
		hom = homogeneity_score(Y, clusters)
		com = completeness_score(Y, clusters)
		sil = silhouette_score(X, clusters)  # Euclidean distance
		aic = gmm_model.aic(X)
		bic = gmm_model.bic(X)

		# Logging metrics
		with open(log_path, 'a') as f:
			out = '{},{},{},{},{},{},{},{}\n'.format(k, time_taken, ari, hom, com, sil, aic, bic)
			f.write(out)


def pca(X, Y):
	print("Running PCA")

	X_train, X_test, Y_train, Y_test = utils.split(X, Y)

	# Run PCA on training data
	pca = PCA()
	pca.fit(X_train)
	eigenvalues = pca.explained_variance_

	# Transform training and testing data
	TX_train = pca.transform(X_train)
	TX_test = pca.transform(X_test)

	# Evaluate on boosting model by incrementally adding components to train
	acc = []
	for i in range(0, len(eigenvalues)):
		model = AdaBoostClassifier(n_estimators=150, learning_rate=0.1)
		model.fit(TX_train[:, :i+1], Y_train)
		Y_pred = model.predict(TX_test[:, :i+1])
		acc.append(accuracy_score(Y_test, Y_pred))

	# Opening log file
	log_path = '../logs/pet_pca.csv'
	with open(log_path, 'w') as f:
		f.write('component,eigenvalues,accuracy\n')

	# Logging metrics
	for i in range(0, len(eigenvalues)):
		with open(log_path, 'a') as f:
			out = '{},{},{}\n'.format(i, eigenvalues[i], acc[i])
			f.write(out)

	# Logging transformed data (first 2 principal components)
	trans_datapath = '../data/pet_pca.csv'
	transformed2 = pd.DataFrame(TX_train[:, :2], columns=['principal1', 'principal2'])
	transformed2['labels'] = Y_train
	transformed2.to_csv(trans_datapath)


def ica(X, Y):
	print("Running ICA")

	X_train, X_test, Y_train, Y_test = utils.split(X, Y)

	n = X.shape[1]  # maximum number of features to get out from ICA
	# Running ICA with incrementing number of components and evaluating kurtosis and accuracy on a model
	avg_kur = []
	acc = []
	for i in range(0, n):
		ica = FastICA(n_components=i+1)
		ica.fit(X_train)

		# Transform training and testing data
		TX_train = ica.transform(X_train)
		TX_test = ica.transform(X_test)

		# Computing kurtosis of training data
		kur = kurtosis(TX_train, 0)     # Already subtracted 3 to get "excess" kurtosis
		avg_kur.append(kur.mean())

		# Evaluate on boosting model
		model = AdaBoostClassifier(n_estimators=150, learning_rate=0.1)
		model.fit(TX_train, Y_train)
		Y_pred = model.predict(TX_test)
		acc.append(accuracy_score(Y_test, Y_pred))

	# Opening log file
	log_path = '../logs/pet_ica.csv'
	with open(log_path, 'w') as f:
		f.write('components,mean_kurtosis,accuracy\n')

	# Logging metrics
	for i in range(0, n):
		with open(log_path, 'a') as f:
			out = '{},{},{}\n'.format(i+1, avg_kur[i], acc[i])
			f.write(out)


def rca(X, Y):
	print("Running RCA")
	iterations = 10  # Number of iterations of rca to perform to average over

	X_train, X_test, Y_train, Y_test = utils.split(X, Y)

	n = X.shape[1]  # maximum number of features to get out from RCA
	# Running RCA with incrementing number of components and evaluating average accuracy on a model
	acc = []
	for i in range(0, n):
		temp_acc = []
		for _ in range(0, iterations):
			rca = GaussianRandomProjection(n_components=i+1)
			rca.fit(X_train)

			# Transform training and testing data
			TX_train = rca.transform(X_train)
			TX_test = rca.transform(X_test)

			# Evaluate on boosting model
			model = AdaBoostClassifier(n_estimators=150, learning_rate=0.1)
			model.fit(TX_train, Y_train)
			Y_pred = model.predict(TX_test)
			temp_acc.append(accuracy_score(Y_test, Y_pred))
		acc.append(np.mean(temp_acc))

	# Opening log file
	log_path = '../logs/pet_rca.csv'
	with open(log_path, 'w') as f:
		f.write('components,accuracy\n')

	# Logging metrics
	for i in range(0, n):
		with open(log_path, 'a') as f:
			out = '{},{}\n'.format(i + 1, acc[i])
			f.write(out)


def rf(X, Y):
	print("Running Random Forest")

	X_train, X_test, Y_train, Y_test = utils.split(X, Y)

	# Run LRF on training data
	rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
	rf.fit(X_train, Y_train)
	importances = rf.feature_importances_

	sorted_impt = np.argsort(importances)[::-1] # Sorted in dec order

	# Evaluate on boosting model by incrementally adding components to train
	acc = []
	for i in range(0, len(importances)):
		col_ix = sorted_impt[:i+1]   # Taking top i features
		model = AdaBoostClassifier(n_estimators=150, learning_rate=0.1)
		model.fit(X_train.iloc[:, col_ix], Y_train)
		Y_pred = model.predict(X_test.iloc[:, col_ix])
		acc.append(accuracy_score(Y_test, Y_pred))

	# Opening log file
	log_path = '../logs/pet_rf_impt.csv'
	with open(log_path, 'w') as f:
		f.write('component,importance\n')

	# Logging metrics
	for i in range(0, len(importances)):
		with open(log_path, 'a') as f:
			out = '{},{}\n'.format(i, importances[i])
			f.write(out)

	# Opening log file
	log_path = '../logs/pet_rf_acc.csv'
	with open(log_path, 'w') as f:
		f.write('components,accuracy\n')

	# Logging metrics
	for i in range(0, len(importances)):
		with open(log_path, 'a') as f:
			out = '{},{}\n'.format(i+1, acc[i])
			f.write(out)


def kmeans_after_red(X, Y):
	print("Running K-means after dimension reduction")

	# Range of k to test
	krange = np.arange(2, 50)
	pca_num_components = 3
	ica_num_components = 2
	rca_num_components = 3
	rf_num_components = 3

	# Dimension Reduction and transforming data to fit
	X_pca, X_ica, X_rca, X_rf = utils.transform4(X, Y, pca_num_components, ica_num_components,
	                                             rca_num_components, rf_num_components)

	# Opening log file
	log_path = '../logs/pet_reduced_kmeans.csv'
	with open(log_path, 'w') as f:
		f.write('k,ari,distortion,pca_ari,pca_distortion,ica_ari,ica_distortion,rca_ari,rca_distortion,rf_ari,rf_distortion\n')

	transformed_X = [X, X_pca, X_ica, X_rca, X_rf]
	for k in krange:
		output = '{}'.format(k)

		for TX in transformed_X:
			# Computing k means
			kmeans_model = KMeans(k, init='random', n_jobs=-1).fit(TX)
			clusters = kmeans_model.predict(TX)

			# Computing metrics
			ari = adjusted_rand_score(Y, clusters)
			dis = kmeans_model.inertia_

			# Appending to output
			output = output + ',{},{}'.format(ari, dis)

		# Logging metrics
		with open(log_path, 'a') as f:
			output = output + '\n'
			f.write(output)


def gmm_after_red(X, Y):
	print("Running GMM after dimension reduction")

	# Range of k to test
	krange = np.arange(2, 50)
	pca_num_components = 3
	ica_num_components = 2
	rca_num_components = 3
	rf_num_components = 3

	# Dimension Reduction and transforming data to fit
	X_pca, X_ica, X_rca, X_rf = utils.transform4(X, Y, pca_num_components, ica_num_components,
	                                             rca_num_components, rf_num_components)

	# Opening log file
	log_path = '../logs/pet_reduced_gmm.csv'
	with open(log_path, 'w') as f:
		f.write(
			'k,ari,aic,bic,pca_ari,pca_aic,pca_bic,ica_ari,ica_aic,ica_bic,rca_ari,rca_aic,rca_bic,rf_ari,rf_aic,rf_bic\n')

	transformed_X = [X, X_pca, X_ica, X_rca, X_rf]
	for k in krange:
		output = '{}'.format(k)

		for TX in transformed_X:
			# Computing k means
			gmm_model = GaussianMixture(k, n_init=10).fit(TX)
			clusters = gmm_model.predict(TX)

			# Computing metrics
			ari = adjusted_rand_score(Y, clusters)
			aic = gmm_model.aic(TX)
			bic = gmm_model.bic(TX)

			# Appending to output
			output = output + ',{},{},{}'.format(ari, aic, bic)

		# Logging metrics
		with open(log_path, 'a') as f:
			output = output + '\n'
			f.write(output)


def main():
	processed_data_path = '../data/processed-pet-outcomes.csv'
	df = pd.read_csv(processed_data_path)
	X = df.iloc[:, :-1]
	Y = df.iloc[:, -1]
	kmeans(X, Y)
	gmm(X, Y)
	pca(X, Y)
	ica(X, Y)
	rca(X, Y)
	rf(X, Y)
	kmeans_after_red(X, Y)
	gmm_after_red(X, Y)


if __name__ == "__main__":
	main()