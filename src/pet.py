import pandas as pd
import numpy as np
from time import clock
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, silhouette_score


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


def main():
	processed_data_path = '../data/processed-pet-outcomes.csv'
	df = pd.read_csv(processed_data_path)
	X = df.iloc[:, :-1]
	Y = df.iloc[:, -1]
	kmeans(X, Y)
	gmm(X, Y)


if __name__ == "__main__":
	main()