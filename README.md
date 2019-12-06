# Anti-aliasing convolutional neural networks for person re-identification
This GitHub repository consists of the code used to perform my master thesis for the Data Science and Society program. 
In this repository the folders 'own_code' and 'models_lpf' consists of functions which are adjusted versions of the work constructed by GitHub user CoinCheung (https://github.com/CoinCheung/triplet-reid-pytorch) which made a replica of the hard triplet loss network proposed by Hermans, Beyer, and Leibe (2017), and the GitHub of Adobe (https://github.com/adobe/antialiased-cnns) for the work on anti-aliasing by Zhang (2019).  

The folder 'results'consists of text files with the results stored of each model discussed in the master thesis.

Besides these folders, four Jupyter Notebooks are stored in this repository.

The Notebook 'create hdf5.ipynb'consists code to load the image data and store the data as numpy arrays in an HDF5 file. This procedure highly decreased the time of loading the data into the Google Colab environment. 

The Notebook 'main_implementation.ipynb' is the main notebook of the experiment and lists code for the preprocessing and the conducting of all trained networks and the evaluation of the overall performance of each network (experiment 3 in the thesis). 

The Notebook 'consistency.ipynb'lists the code used to conduct experiment 1 in the thesis. In this notebook the consistency measure is created and tested. 

The Notebook 'performance_fooled_networks.ipynb' lists the code which is used to conduct experiment 2 in the thesis. In this experiment the anti-aliased networks are tested when the input is shifted. 

The Notebook 're-ranking.ipynb'lists the code which is used to re-rank the output of the models as described in experiment 4 in the thesis. 
