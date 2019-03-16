import pandas as pd
import matplotlib.pyplot as plt


def plot_kmeans():
	log_filepath = '../logs/pet_kmeans.csv'
	base_graphpath = '../graphs/pet_kmeans_@plot@.png'
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
	log_filepath = '../logs/pet_gmm.csv'
	base_graphpath = '../graphs/pet_gmm_@plot@.png'
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
	plt.title("GMM: k vs internal metrics (silhouette)")
	plt.xlabel('k')
	plt.ylabel('Score')
	sil = data['silhouette'] / data['silhouette'].max()
	plt.plot(x, sil, label='Normalized Silhouette')
	aic = data['aic'] / data['aic'].max()
	plt.plot(x, aic, label='Normalized AIC')
	bic = data['bic'] / data['bic'].max()
	plt.plot(x, bic, label='Normalized BIC')
	plt.legend()
	plt.savefig(base_graphpath.replace('@plot@', 'int'), dpi=300)


def plot_kmeans_gmm_comparison():
	log_kmeans_filepath = '../logs/pet_kmeans.csv'
	log_gmm_filepath = '../logs/pet_gmm.csv'
	base_graphpath = '../graphs/pet_cluster_comp_@plot@.png'

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


def main():
	plot_kmeans()
	plot_gmm()
	plot_kmeans_gmm_comparison()
	plt.show()


if __name__ == "__main__":
	main()