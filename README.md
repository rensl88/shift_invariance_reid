# Anti-aliasing convolutional neural networks for person re-identification
This GitHub repository consists of the code used to perform my master thesis for the Data Science and Society program. 
In this repository the folders 'own_code' and 'models_lpf' consists of functions which are adjusted versions of the work constructed by GitHub user CoinCheung (https://github.com/CoinCheung/triplet-reid-pytorch) which made a replica of the hard triplet loss network proposed by Hermans, Beyer, and Leibe (2017), and the GitHub of Adobe (https://github.com/adobe/antialiased-cnns) for the work on anti-aliasing by Zhang (2019).

## Person Re-Identification 
Person Re-Identification is about identifying a person across different cameras with different angles and positions. 
Specificly the task is to match a particular query image of a person to images of the same identity out of a big gallery dataset that contains multiple images in a wide range of people. 

In this field, a common approach is to use convolutional neural networks with three streams, a query stream (the one of the image itself) a negative stream (a stream with the image of a person with a different identity than the query image) and a positive stream (a stream with the image of a person with the same identity as the query image. Here the network is optimized with hard triplet loss, to maximize the distance between the query images and the negative examples and to minimize the distance between the query images and the positive examples. The final pipeline for the model architecture looks as follows:

![Image description](https://github.com/rensl88/shift_invariance_reid/blob/master/images/architecture.png)

Anti-aliasing is a technique used in signal processing to smoothen signals. In CNNs for image recognition tasks the method has proven to be an effective way to make a network less variant to shifts in the input. It improved the so-called shift invariance of networks. To do so, the network uses maxpooling with a stride of 1 to first remain most information in the image and then afterwards it uses a blur filter (of the size 2,3 or 5) as shown in the image below. 
![Image description](https://github.com/rensl88/shift_invariance_reid/blob/master/images/antialiasing.png)

In the experiments belonging to the study it is tested whether the method called anti-aliasing has an effect on first the shift-invariance of the networks and besides that on the overall accuracy of the network. 

The first experiment tested the consistency of an anti-aliased network, compared to a baseline model having no anti-aliasing applied to the network. The consistency is a measure constructed to calculate whether a network outputs the same output for images without a slight vertical and horizontal shift and images with slights shifts. This experiment showed that anti-aliased methods indeed improve the shift-invariance of CNNs for person re-identification, the measured consistency is shown in the table below:
![Image description](https://github.com/rensl88/shift_invariance_reid/blob/master/images/consistency.png)

The second experiment tested the general accuracy of anti-aliased networks. This showed that despite the fact that the anti-aliasing networks are more robust to shifts in the input, this decreased the performance of the network. This can be because the blur filter serves as a low-pass filter which causes information to be lost. The accuracy is of networks is shown in the table below.

To improve CNNs for person re-identification, almost in every approach the output of the three streamed architecture is re-ranked. This is done by using an adjusted nearest neighbour approach. The outcomes show that both the baseline model and the anti-aliased model gain accuracy when re-ranking is applied. 
![Image description](https://github.com/rensl88/shift_invariance_reid/blob/master/images/accuracy.png)

More details about the experiments and the theory are included in the master thesis. Access can be requested trough the following link: https://drive.google.com/file/d/16RI3WgMKv0EXNtJ4Cv5FrP9qnj_kpYZ4/view?usp=sharing


## Overview of notebooks
The folder 'results' consists of text files with the results stored of each model discussed in the master thesis.

Besides these folders, six Jupyter Notebooks are stored in this repository.

The Notebook 'create_hdf5_.ipynb' consists code to load the image data and store the data as numpy arrays in an HDF5 file. This procedure highly decreased the time of loading the data into the Google Colab environment. 

The Notebook 'main_implementation.ipynb' is the main notebook of the experiment and lists code for the preprocessing and the conducting of all trained networks and the evaluation of the overall performance of each network (experiment 3 in the thesis). 

The Notebook 'consistency.ipynb'lists the code used to conduct experiment 1 in the thesis. In this notebook the consistency measure is created and tested. 

The Notebook 'performance_fooled_networks.ipynb' lists the code which is used to conduct experiment 2 in the thesis. In this experiment the anti-aliased networks are tested when the input is shifted. 

The Notebook 're-ranking.ipynb'lists the code which is used to re-rank the output of the models as described in experiment 4 in the thesis. 

The Notebook 'visualizing_cmc_ranking,ipynb' consists of code used to create the cmc plot. 
