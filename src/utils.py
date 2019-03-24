import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import RandomForestClassifier


def split(X, Y, test_frac=0.1):
	# Split data
	cutoff = int(test_frac * len(Y))
	ix = np.arange(0, len(Y))
	np.random.shuffle(ix)
	test_index = ix[:cutoff]
	train_index = ix[cutoff:]
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

	return X_train, X_test, Y_train, Y_test


def transform4(X, Y, pca_num, ica_num, rca_num, rf_num):
	# Dimension Reduction and transforming data to fit
	pca = PCA(n_components=pca_num)
	X_pca = pca.fit_transform(X)

	ica = FastICA(n_components=ica_num)
	X_ica = ica.fit_transform(X)

	rca = GaussianRandomProjection(n_components=rca_num)
	X_rca = rca.fit_transform(X)

	rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
	rf.fit(X, Y)
	importances = rf.feature_importances_
	sorted_impt = np.argsort(importances)[::-1]  # Sorted in dec order
	X_rf = X.iloc[:, sorted_impt[:rf_num]]

	return X_pca, X_ica, X_rca, X_rf