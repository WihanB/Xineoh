import tensorflow as tf
import numpy as np
import queue_maker
import time
file_name = 'TF_mnist_train'

def make_example(pixels,label):
    ##This function makes the example one at a time that will be written to tensorflow records file
    ex =tf.train.Example()
    ##add in the pixels
    fl_pixels = ex.features.feature["pixels"]
    fl_labels = ex.features.feature["labels"]

    [fl_pixels.float_list.value.append(pixel) for pixel in pixels]
    fl_labels.int64_list.value.append(label)

    return ex


def write_example(writer,example):
    ###writer is the tf writing object and the example is the output of make example
    writer.write(example.SerializeToString())
    return 1

def main():
    with tf.Graph().as_default():
        # inputs
        sess=tf.Session()
        train_x,train_t=queue_maker.inputs_biased(1,10000,file_name)

        writer = tf.python_io.TFRecordWriter('TF_mnist_train_ub')
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        start_time = time.time()
        for i in range(240000):
            pixels,label=sess.run([train_x,train_t])
            example=make_example(pixels[0].tolist()[0],label[0].tolist())
            write_example(writer,example)
            if i % 1000==0:
                end_time=time.time()
                elapsed=end_time-start_time
                print('%d Samples of 240000 written in %.3f Seconds'%(i,elapsed))
                start_time=time.time()


        writer.close()
        coord.request_stop()
        coord.request_stop()
        coord.join(threads)
        sess.close()
        sess.close()

if __name__ == '__main__':
    main()


