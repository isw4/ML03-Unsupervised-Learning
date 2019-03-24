import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_kmeans():
	log_filepath = '../logs/wine_kmeans.csv'
	base_graphpath = '../graphs/wine_kmeans_@plot@.png'
	data = pd.read_csv(log_filepath)

	x = data['k']

	# Time
	plt.figure()
	plt.title("K-means: k vs time")
	plt.xlabel('k')
	plt.ylabel('Time (s)')
	time = data['time']
	plt.plot(x, time)
	plt.savefig(base_graphpath.replace('@plot@', 'time'), dpi=300)

	# External metrics (ARI, homogeneity, completeness)
	plt.figure()
	plt.title("K-means: k vs external metrics")
	plt.xlabel('k')
	plt.ylabel('Score')
	ari = data['ari']
	plt.plot(x, ari, label='ARI')
	hom = data['homogeneity']
	plt.plot(x, hom, label='Homogeneity')
	com = data['completeness']
	plt.plot(x, com, label='Completeness')
	plt.legend()
	plt.savefig(base_graphpath.replace('@plot@', 'ext'), dpi=300)

	# Internal metrics (silhouette, distortion)
	plt.figure()
	plt.title("K-means: k vs internal metrics")
	plt.xlabel('k')
	plt.ylabel('Score')
	sil = data['silhouette'] / data['silhouette'].max()
	plt.plot(x, sil, label='Normalized Silhouette')
	dis = data['distortion'] / data['distortion'].max()
	plt.plot(x, dis, label='Normalized Distortion')
	plt.legend()
	plt.savefig(base_graphpath.replace('@plot@', 'int'), dpi=300)

	# Combined metrics (ARI, distortion)
	plt.figure()
	plt.title("K-means: k vs ARI and distortion")
	plt.xlabel('k')
	plt.ylabel('Score')
	dis = data['distortion'] / data['distortion'].max()
	plt.plot(x, dis, label='Normalized Distortion')
	ari = data['ari'] / data['ari'].max()
	plt.plot(x, ari, label='Normalized ARI')
	plt.legend()
	plt.savefig(base_graphpath.replace('@plot@', 'combined'), dpi=300)


def plot_gmm():
	log_filepath = '../logs/wine_gmm.csv'
	base_graphpath = '../graphs/wine_gmm_@plot@.png'
	data = pd.read_csv(log_filepath)

	x = data['k']

	# Time
	plt.figure()
	plt.title("GMM: k vs time")
	plt.xlabel('k')
	plt.ylabel('Time (s)')
	time = data['time']
	plt.plot(x, time)
	plt.savefig(base_graphpath.replace('@plot@', 'time'), dpi=300)

	# External metrics (ARI, homogeneity, completeness)
	plt.figure()
	plt.title("GMM: k vs external metrics")
	plt.xlabel('k')
	plt.ylabel('Score')
	ari = data['ari']
	plt.plot(x, ari, label='ARI')
	hom = data['homogeneity']
	plt.plot(x, hom, label='Homogeneity')
	com = data['completeness']
	plt.plot(x, com, label='Completeness')
	plt.legend()
	plt.savefig(base_graphpath.replace('@plot@', 'ext'), dpi=300)

	# Internal metrics (silhouette, AIC, BIC)
	plt.figure()
	plt.title("GMM: k vs internal metrics")
	plt.xlabel('k')
	plt.ylabel('Score')
	sil = data['silhouette'] / data['silhouette'].max()
	plt.plot(x, sil, label='Normalized Silhouette')
	aic = data['aic'] / data['aic'].abs().max()
	plt.plot(x, aic, label='Normalized AIC')
	bic = data['bic'] / data['bic'].abs().max()
	plt.plot(x, bic, label='Normalized BIC')
	plt.legend()
	plt.savefig(base_graphpath.replace('@plot@', 'int'), dpi=300)


def plot_kmeans_gmm_comparison():
	log_kmeans_filepath = '../logs/wine_kmeans.csv'
	log_gmm_filepath = '../logs/wine_gmm.csv'
	base_graphpath = '../graphs/wine_cluster_comp_@plot@.png'

	kmeans = pd.read_csv(log_kmeans_filepath)
	gmm = pd.read_csv(log_gmm_filepath)

	x_kmeans = kmeans['k']
	x_gmm = gmm['k']

	# External metrics (ARI)
	plt.figure()
	plt.title("K-means and GMM: k vs external metrics (ARI)")
	plt.xlabel('k')
	plt.ylabel('Score')
	ari_kmeans = kmeans['ari']
	plt.plot(x_kmeans, ari_kmeans, label='K-means')
	ari_gmm = gmm['ari']
	plt.plot(x_gmm, ari_gmm, label='GMM')
	plt.legend()
	plt.savefig(base_graphpath.replace('@plot@', 'ari'), dpi=300)

	# Internal metrics (silhouette)
	plt.figure()
	plt.title("K-means and GMM: k vs internal metrics (distortion, BIC)")
	plt.xlabel('k')
	plt.ylabel('Score')
	sil_kmeans = kmeans['silhouette']
	plt.plot(x_kmeans, sil_kmeans, label='K-means')
	sil_gmm = gmm['silhouette']
	plt.plot(x_gmm, sil_gmm, label='GMM')
	plt.legend()
	plt.savefig(base_graphpath.replace('@plot@', 'sil'), dpi=300)


def plot_pca():
	log_pca_filepath = '../logs/wine_pca.csv'
	base_graphpath = '../graphs/wine_pca.png'
	pca = pd.read_csv(log_pca_filepath)

	eigenvalues = pca['eigenvalues'].values
	acc = pca['accuracy'].values

	# Compute cumulative variance explained
	cum_variance = [eigenvalues[0]]
	for i in range(1, len(eigenvalues)):
		cum_variance.append(cum_variance[-1] + eigenvalues[i])
	cum_variance = np.array(cum_variance)

	# Normalize variance
	cum_variance = cum_variance / cum_variance.max()

	# Plot all
	components = np.arange(0, len(eigenvalues)) + 1
	plt.figure()
	plt.plot(components, cum_variance, label='Cumulative variance explained')
	plt.plot(components, acc, label='Accuracy')
	plt.title('Principle Component Analysis (Wine)')
	plt.xlabel('Number of components')
	plt.ylabel('Normalized values')
	plt.xticks(components)
	plt.grid(True)
	plt.legend()
	plt.savefig(base_graphpath, dpi=300)


def plot_ica():
	log_ica_filepath = '../logs/wine_ica.csv'
	base_graphpath = '../graphs/wine_ica.png'
	ica = pd.read_csv(log_ica_filepath)

	components = ica['components'].values
	kur = ica['mean_kurtosis'].values
	acc = ica['accuracy'].values

	# Plot all
	fig, ax1 = plt.subplots()
	plt.grid(True)

	color = 'tab:red'
	ax1.set_title('Independent Component Analysis (Wine)')
	ax1.set_xlabel('Number of components')
	ax1.set_ylabel('Mean Kurtosis', color=color)
	ax1.plot(components, kur, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('Accuracy score', color=color)  # we already handled the x-label with ax1
	ax2.plot(components, acc, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	plt.xticks(components)
	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.savefig(base_graphpath, dpi=300)


def plot_rca():
	log_rca_filepath = '../logs/wine_rca.csv'
	base_graphpath = '../graphs/wine_rca.png'
	rca = pd.read_csv(log_rca_filepath)

	components = rca['components'].values
	acc = rca['accuracy'].values

	plt.figure()
	plt.title('Random Projections (Wine)')
	plt.xlabel('Number of components')
	plt.ylabel('Mean Accuracy')
	plt.plot(components, acc)
	plt.xticks(components)
	plt.grid(True)
	plt.savefig(base_graphpath, dpi=300)


def plot_rf():
	log_rf_filepath1 = '../logs/wine_rf_impt.csv'
	log_rf_filepath2 = '../logs/wine_rf_acc.csv'
	base_graphpath = '../graphs/wine_rf.png'
	rf_impt = pd.read_csv(log_rf_filepath1)
	rf_acc = pd.read_csv(log_rf_filepath2)

	importance = rf_impt['importance'].values
	acc = rf_acc['accuracy'].values

	# Compute cumulative importance after sorting
	sorted_impt = np.sort(importance)[::-1]
	cum_impt = [sorted_impt[0]]
	for i in range(1, len(sorted_impt)):
		cum_impt.append(cum_impt[-1] + sorted_impt[i])
	cum_impt = np.array(cum_impt)

	# Normalize eigenvalues and variance
	cum_impt = cum_impt / cum_impt.max()

	# Plot all
	components = np.arange(0, len(importance)) + 1
	plt.figure()
	plt.plot(components, cum_impt, label='Cumulative importance')
	plt.plot(components, acc, label='Accuracy')
	plt.title('Random Forest (Wine)')
	plt.xlabel('Number of components')
	plt.ylabel('Normalized values')
	plt.xticks(components)
	plt.grid(True)
	plt.legend()
	plt.savefig(base_graphpath, dpi=300)


def plot_reduced_kmeans():
	log_redkmeans_filepath = '../logs/wine_reduced_kmeans.csv'
	base_graphpath = '../graphs/wine_reduced_kmeans_@plot@.png'
	kmeans = pd.read_csv(log_redkmeans_filepath)

	x = kmeans['k'].values + 1

	plt.figure()
	plt.title("K-means after dimension reduction (ARI)")
	plt.xlabel('k')
	plt.ylabel('Score')
	ari = kmeans['ari']
	plt.plot(x, ari, label='No reduction')
	pca_ari = kmeans['pca_ari']
	plt.plot(x, pca_ari, label='PCA')
	ica_ari = kmeans['ica_ari']
	plt.plot(x, ica_ari, label='ICA')
	rca_ari = kmeans['rca_ari']
	plt.plot(x, rca_ari, label='RP')
	rf_ari = kmeans['rf_ari']
	plt.plot(x, rf_ari, label='RF')
	plt.legend()
	plt.savefig(base_graphpath.replace('@plot@', 'ari'), dpi=300)

	plt.figure()
	plt.title("K-means after dimension reduction (distortion)")
	plt.xlabel('k')
	plt.ylabel('Score')
	dis = kmeans['distortion']
	plt.plot(x, dis, label='No reduction')
	pca_dis = kmeans['pca_distortion']
	plt.plot(x, pca_dis, label='PCA')
	ica_dis = kmeans['ica_distortion']
	plt.plot(x, ica_dis, label='ICA')
	rca_dis = kmeans['rca_distortion']
	plt.plot(x, rca_dis, label='RP')
	rf_dis = kmeans['rf_distortion']
	plt.plot(x, rf_dis, label='RF')
	plt.legend()
	plt.savefig(base_graphpath.replace('@plot@', 'dis'), dpi=300)

	plt.figure()
	plt.title("K-means after dimension reduction (distortion)")
	plt.xlabel('k')
	plt.ylabel('Normalized Score')
	dis = kmeans['distortion'] / kmeans['distortion'].max()
	plt.plot(x, dis, label='No reduction')
	pca_dis = kmeans['pca_distortion'] / kmeans['pca_distortion'].max()
	plt.plot(x, pca_dis, label='PCA')
	ica_dis = kmeans['ica_distortion'] / kmeans['ica_distortion'].max()
	plt.plot(x, ica_dis, label='ICA')
	rca_dis = kmeans['rca_distortion'] / kmeans['rca_distortion'].max()
	plt.plot(x, rca_dis, label='RP')
	rf_dis = kmeans['rf_distortion'] / kmeans['rf_distortion'].max()
	plt.plot(x, rf_dis, label='RF')
	plt.legend()
	plt.savefig(base_graphpath.replace('@plot@', 'norm_dis'), dpi=300)


def plot_reduced_gmm():
	log_redgmm_filepath = '../logs/wine_reduced_gmm.csv'
	base_graphpath = '../graphs/wine_reduced_gmm_@plot@.png'
	gmm = pd.read_csv(log_redgmm_filepath)

	x = gmm['k'].values + 1

	plt.figure()
	plt.title("GMM after dimension reduction (ARI)")
	plt.xlabel('k')
	plt.ylabel('Score')
	ari = gmm['ari']
	plt.plot(x, ari, label='No reduction')
	pca_ari = gmm['pca_ari']
	plt.plot(x, pca_ari, label='PCA')
	ica_ari = gmm['ica_ari']
	plt.plot(x, ica_ari, label='ICA')
	rca_ari = gmm['rca_ari']
	plt.plot(x, rca_ari, label='RP')
	rf_ari = gmm['rf_ari']
	plt.plot(x, rf_ari, label='RF')
	plt.legend()
	plt.savefig(base_graphpath.replace('@plot@', 'ari'), dpi=300)

	plt.figure()
	plt.title("GMM after dimension reduction (AIC)")
	plt.xlabel('k')
	plt.ylabel('Score')
	aic = gmm['aic']
	plt.plot(x, aic, label='No reduction')
	pca_aic = gmm['pca_aic']
	plt.plot(x, pca_aic, label='PCA')
	ica_aic = gmm['ica_aic']
	plt.plot(x, ica_aic, label='ICA')
	rca_aic = gmm['rca_aic']
	plt.plot(x, rca_aic, label='RP')
	rf_aic = gmm['rf_aic']
	plt.plot(x, rf_aic, label='RF')
	plt.legend()
	plt.savefig(base_graphpath.replace('@plot@', 'aic'), dpi=300)

	plt.figure()
	plt.title("K-means after dimension reduction (BIC)")
	plt.xlabel('k')
	plt.ylabel('Score')
	bic = gmm['bic']
	plt.plot(x, bic, label='No reduction')
	pca_bic = gmm['pca_bic']
	plt.plot(x, pca_bic, label='PCA')
	ica_bic = gmm['ica_bic']
	plt.plot(x, ica_bic, label='ICA')
	rca_bic = gmm['rca_bic']
	plt.plot(x, rca_bic, label='RP')
	rf_bic = gmm['rf_bic']
	plt.plot(x, rf_bic, label='RF')
	plt.legend()
	plt.savefig(base_graphpath.replace('@plot@', 'bic'), dpi=300)


def main():
	plot_kmeans()
	plot_gmm()
	plot_kmeans_gmm_comparison()
	plot_pca()
	plot_ica()
	plot_rca()
	plot_rf()
	plot_reduced_kmeans()
	plot_reduced_gmm()
	plt.show()


if __name__ == "__main__":
	main()