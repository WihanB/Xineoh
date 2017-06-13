##This is the main file to be run that executes training, loads data and outputs results
import os
import Mnist_Data_Reader
import mnist_dash
import Mnist_resampler
filepath="./TF_mnist_train"
filepath_resamp="./TF_mnist_train_ub"
weights_path="./my_weights.meta"
if not os.path.exists(filepath):
    print("Fetching Data From SQL Server")
    Mnist_Data_Reader.data_creator()

if not os.path.exists(filepath_resamp):
    print("Sampling Out Class Imbalances")
    Mnist_resampler.main()

if not os.path.exists(weights_path):
    print("Training Deep Conv Net")
    mnist_dash.run_training(Training=True)
else:
    mnist_dash.run_training(Training=False)
