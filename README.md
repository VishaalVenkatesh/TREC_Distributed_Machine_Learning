## Overview

The Text REtrieval Conference (TREC) conference is  an annual conference held to facilitate research in analysing social media data to help emergency responders. Specifically, the
TREC Incident Streams (2020) Incident Streams is a TREC track designed to bring together academia and industry to research technologies to automaticaly process social media streams 
during emergency situations with the aim of categorizing information and aid requests made on social media for emergency service operators.

**TREC Website** - http://dcs.gla.ac.uk/~richardm/TREC_IS/

## About the Data

We have at our disposal four different datasets. A train dataset from 2018, a test dataset from 2018 and 2 different test datasets from 2019. The tweets are classified into four 
different priority categories - low, medium, high and critical. Our research specifically focuses on identifying and classifying the crtitical tweets from the rest. It has to be 
noted that the critical tweets are far and few and account for less than **1%** of all tweets. Because of the sever imbalance of the dataset researchers have faced difficulties 
when attempting to classify critical tweets. We aim to make progress in this avenue. 

## Our Goals
1. Compare different feature extraction methods - TFIDF, word embeddings (Word2Vec, GloVe, Keras Embedding Layer) etc. 
2. Develop a classifier to accurately identify critical tweets with resaonable precision and recall. We are exploring SVMs and LSTM RNNs.
3. Explore suitable optimizers (with an emphasis on 2nd order quasi-Newton methods) for deployment of the model in a distributed setting.
4. Intergrate the optimisation method with GADGET SVMs to study performance.
5. Finally, explore ways to integrate deep learning method (LSTM RNN) in a distributed setting and evaluate performance.

## Progress So Far...

SVM models perform exceedigly well when TFIDF features are used. This is because SVMs are known to perform well when the feature space is large. Neural networks do not train on TFIDF very well. This is because TFIDF are a sparse form of representation with many empty entries. We have not attempted to reduce the feature dimensionalty using PCA, t-SNE or 
manifold learning but this is something we could look into in thw future. As far as embeddings are concerned, keras's built-in embedding layer performs well when couple with an LSTM RNN. However, Word2Vec embeddings work well when trained using 1D convolutional layers. We are yet to explore GLoVe embeddings.
