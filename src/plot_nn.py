import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
	log_redgmm_filepath = '../logs/wine_reduced_gmm.csv'
	base_graphpath = '../graphs/wine_reduced_gmm_@plot@.png'
	gmm = pd.read_csv(log_redgmm_filepath)


if __name__ == "__main__":
	main()