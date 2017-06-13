# Xineoh
Created for Xineoh Challenge

For this to run, V1.0+ of tensorflow is required, for an installation guide please see: https://www.tensorflow.org/install/

In addition, training on CPU can take very long, It is suggested that training is done on a GPU.

The main file which will produce the confusion matrix and print the output accuracy is named Xineoh_Predictor.py

Running this file will fetch the data from the SQL server, create local storage of the data in a TFrecords file
Please Note: TF records files are rather verbose, as a result they use quite a lot of storage.

After this has been run, the data will be resampled in order to produce a new training set which solves the class imbalance problem
Once again this is a very large file (700mb) This is probably the slowest part of the process due to an inefficiency in tensorflow's resampling code (I have edited in in mine but it will still be slow in your installation) . The reason these files have been created is that in the case where we are not solving a toy problem such as MNIST, it will be possible to still train a model without running OOM. You will notice that the predictor is not memory intensive at all.

Once the data has been resampled the Convolutional Neural Network will be trained and the output produced, the model has an accuracy of around 99.1% based on initialisation
