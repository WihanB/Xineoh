#!/usr/bin/python
# Get Xineoh's Mnist Data
import mysql.connector as mysql
import time
import tensorflow as tf
import numpy as np

hostname = '173.45.91.18'
username = 'user01'
password = 'nw21ZDcWzTG4'
database = 'mnist01'
###Specify some preliminaries like what we want the output files to be called
train_out = 'TF_mnist_train'
test_out = 'TF_mnist_test'
count = np.zeros([10, 1])
###Specify what will exist in these records files (labels and data)
x_vals = 'raw_pixels'
t_vals = 'labels'


# ##One approach to the class imbalance is to resample the data
# ##to obtain a uniform distribution, since there are a large
# #number of samples this should be sufficient
# def resample(probabilities,samples,Nsamples):
#     #arguments should be np arrays
#     cumulative_p=np.cumsum(probabilities)
#     new_probs=np.random.random([Nsamples,1])
#     newsamp=np.zeros_like(samples)*np.max(cumulative_p)
#     idx=np.zeros_like(probabilities)



def grab_max(train_dataset, cnx):
    ## gets the max index of the SQL table
    cursor = cnx.cursor(buffered=True)
    if train_dataset == True:
        selected_table = "mnist_train"
    else:
        selected_table = "mnist_test"

    cursor.execute("SELECT MAX(id) FROM " + selected_table)
    rows = cursor.fetchone()
    cursor.close()
    print(rows[0])
    return int(rows[0])


def grab_example(train_dataset, cnx, idx):
    ##gets the max index of the SQL table
    label_list = []
    pixel_list = []
    cursor = cnx.cursor(buffered=True)
    if train_dataset == True:
        selected_table = "mnist_train"
    else:
        selected_table = "mnist_test"
    start = time.time()
    cursor.execute("SELECT data FROM " + selected_table + " WHERE id<=%d;" % idx)
    print(time.time() - start)
    rows_out = cursor.fetchall()

    for i in range(idx):
        rows = rows_out[i][0]
        data_out = list(map(int, rows.split(',')))
        label = data_out[0]
        pixels = data_out[1:]
        pixels = list(map(float, pixels))
        pixel_list.append(pixels)
        label_list.append(label)
    cursor.close()
    # print(pixels)
    # count[label]+=1
    # print(count/float(sum(count)))
    return label_list, pixel_list


# Simple routine to run a query on a database and print the results:


##
def make_example(pixels, label):
    # This function makes the example one at a time that will be written to tensorflow records file
    ex = tf.train.Example()
    ##add in the pixels
    fl_pixels = ex.features.feature["pixels"]
    fl_labels = ex.features.feature["labels"]

    [fl_pixels.float_list.value.append(pixel) for pixel in pixels]
    fl_labels.int64_list.value.append(label)

    return ex


def write_example(writer, example):
    ###writer is the tf writing object and the example is the output of make example
    writer.write(example.SerializeToString())
    return 1


def fill_record_file(train_dataset, max_examples, cnx):
    if train_dataset == True:
        filename = train_out
        maxval = max_examples['train']
    else:
        filename = test_out
        maxval = max_examples['test']
    writer = tf.python_io.TFRecordWriter(filename)
    with tf.Session() as sess:
        ##Starts my queue runners to ensure that threads start and stop when i want them to
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        label_list, pixel_list = grab_example(train_dataset, cnx, maxval)
        for i in range(maxval):
            lab, pix = label_list[i], pixel_list[i]
            if train_dataset == True:
                count[lab] += 1
                print(count / np.sum(count))

            example = make_example(pix, lab)
            write_example(writer, example)

        writer.close()
        coord.request_stop()
        coord.request_stop()
        coord.join(threads)
        sess.close()
        sess.close()


def data_creator():
    cnx = mysql.connect(host=hostname, user=username, passwd=password, db=database)
    max_train = grab_max(True, cnx)
    max_test = grab_max(False, cnx)
    feed_dict = {'train': max_train, 'test': max_test}
    fill_record_file(True, feed_dict, cnx)
    fill_record_file(False, feed_dict, cnx)
    cnx.close()


if __name__ == '__main__':
    data_creator()
