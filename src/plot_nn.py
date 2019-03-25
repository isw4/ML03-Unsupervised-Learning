import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
	log_filepath = '../logs/wine_nn.csv'
	base_graphpath = '../graphs/wine_nn@plot@.png'
	data = pd.read_csv(log_filepath)

	## Time plot
	time = data['time'][0]
	pca_time = data['pca_time'][0]
	ica_time = data['ica_time'][0]
	rca_time = data['rca_time'][0]
	rf_time = data['rf_time'][0]
	pcak_time = data['pca_km_time'][0]
	icak_time = data['ica_km_time'][0]
	rcak_time = data['rca_km_time'][0]
	rfk_time = data['rf_km_time'][0]
	pcag_time = data['pca_gmm_time'][0]
	icag_time = data['ica_gmm_time'][0]
	rcag_time = data['rca_gmm_time'][0]
	rfg_time = data['rf_gmm_time'][0]

	# set width of bar
	barWidth = 0.25

	# set height of bar
	ctl_bar = [time]
	red_bar = [pca_time, ica_time, rca_time, rf_time]
	km_bar = [pcak_time, icak_time, rcak_time, rfk_time]
	gmm_bar = [pcag_time, icag_time, rcag_time, rfg_time]

	# Set position of bar on X axis
	r0 = [0]
	r1 = np.arange(0, len(red_bar)) + 0.75
	r2 = [x + barWidth for x in r1]
	r3 = [x + barWidth for x in r2]

	# Make the plot
	plt.figure()
	plt.title('Time taken for neural network training (including various data processing)')
	plt.bar(r0, ctl_bar, color='#587181', width=barWidth, edgecolor='white', label='Control')
	plt.bar(r1, red_bar, color='#7ed9b7', width=barWidth, edgecolor='white', label='Only Dim. Reduction')
	plt.bar(r2, km_bar, color='#ff6f61', width=barWidth, edgecolor='white', label='DR + K-means')
	plt.bar(r3, gmm_bar, color='#0392cf', width=barWidth, edgecolor='white', label='DR + GMM')

	# Add xticks on the middle of the group bars
	plt.ylabel('Time (seconds)')
	plt.xlabel('Dimension Reduction Algorithm')
	plt.xticks([r for r in range(0, 5)], ['Original', 'PCA', 'ICA', 'RP', 'RF'])

	# Create legend & Show graphic
	plt.legend(loc='upper right')
	plt.savefig(base_graphpath.replace('@plot@', 'time'), dpi=300)

	## Bar plot of accuracy
	test = data['test_accuracy'][0]
	pca_test = data['pca_test_accuracy'][0]
	ica_test = data['ica_test_accuracy'][0]
	rca_test = data['rca_test_accuracy'][0]
	rf_test = data['rf_test_accuracy'][0]
	k_test = data['km_test_accuracy'][0]
	pcak_test = data['pca_km_test_accuracy'][0]
	icak_test = data['ica_km_test_accuracy'][0]
	rcak_test = data['rca_km_test_accuracy'][0]
	rfk_test = data['rf_km_test_accuracy'][0]
	g_test = data['gmm_test_accuracy'][0]
	pcag_test = data['pca_gmm_test_accuracy'][0]
	icag_test = data['ica_gmm_test_accuracy'][0]
	rcag_test = data['rca_gmm_test_accuracy'][0]
	rfg_test = data['rf_gmm_test_accuracy'][0]

	# set width of bar
	barWidth = 0.25

	# set height of bar
	ctl_bar = [test]
	red_bar = [pca_test, ica_test, rca_test, rf_test]
	km_bar   = [pcak_test, icak_test, rcak_test, rfk_test]
	gmm_bar = [pcag_test, icag_test, rcag_test, rfg_test]

	# Set position of bar on X axis
	r0 = [0]
	r1 = np.arange(0, len(red_bar)) + 0.75
	r2 = [x + barWidth for x in r1]
	r3 = [x + barWidth for x in r2]

	# Make the plot
	plt.figure()
	plt.title('Accuracy on neural network after various data processing')
	plt.bar(r0, ctl_bar, color='#587181', width=barWidth, edgecolor='white', label='Control')
	plt.bar(r1, red_bar, color='#7ed9b7', width=barWidth, edgecolor='white', label='Only Dim. Reduction')
	plt.bar(r2, km_bar, color='#ff6f61', width=barWidth, edgecolor='white', label='DR + K-means')
	plt.bar(r3, gmm_bar, color='#0392cf', width=barWidth, edgecolor='white', label='DR + GMM')

	# Add xticks on the middle of the group bars
	plt.ylabel('Accuracy Score')
	plt.xlabel('Dimension Reduction Algorithm')
	plt.xticks([r for r in range(0, 5)], ['Original', 'PCA', 'ICA', 'RP', 'RF'])

	# Create legend & Show graphic
	plt.legend(loc='lower right')
	plt.savefig(base_graphpath.replace('@plot@', ''), dpi=300)
	plt.show()


if __name__ == "__main__":
	main()