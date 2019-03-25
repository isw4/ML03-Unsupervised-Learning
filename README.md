### CS 7641 Assignment 3 - Unsupervised Learning and Dimensionality Reduction

Code and data can be found at: https://github.com/isw4/ML03-Unsupervised-Learning

## Directories
**src/**  
    Contains the source code  
**data/**  
    Contains the raw and cleaned csv data files, along with some description of each set  
**logs/**  
    Contains data output after running the experiments  
**graphs/**  
    Contains graphs that are generated from the logs


## Setup

1)  Make sure to have Conda installed

2)  Install the conda environment:
    ~~~
    conda env create -f environment.yml
    ~~~

3)  Activate the environment:
    If using Windows, open the Anaconda prompt and enter:
    ~~~
    activate ml3
    ~~~

    If using Mac or Linux, open the terminal and enter:
    ~~~
    source activate ml3
    ~~~


## Running the datasets

From the root directory:  

To make fresh processed csv data from the raw data(for both wine and pet datasets):
~~~
python ./src/data_processing.py
~~~

To run the experiments:
~~~
python ./src/wine.py
python ./src/pet.py
python ./src/nn.py
~~~

The data from the experiments will be saved into the ./logs/ directory. You can then plot the data with:
~~~
python ./src/plot_wine.py
python ./src/plot_pet.py
python ./src/plot_nn.py
~~~