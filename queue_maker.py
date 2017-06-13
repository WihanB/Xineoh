import tensorflow as tf


def read_and_decode(filename,scaling=True):
    # This function is called by the input to read from the file and feed data

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename)

    features={
    'pixels' : tf.FixedLenFeature([784],tf.float32),
    'labels' : tf.FixedLenFeature([],tf.int64)
    }

    my_examples=tf.parse_single_example(serialized_example,features=features)
    with tf.name_scope('Preporcessing'):
        #cast into appropriate Dtypes
        pixel_values=tf.cast(my_examples['pixels'],tf.float32)
        label_values=tf.cast(my_examples['labels'],tf.int32)
        if scaling==True:
            with tf.name_scope('scaling'):
                #scale the images between -0.5 and 0.5
                pixel_values=tf.subtract(tf.divide(pixel_values,255.0),0.5)

    return pixel_values, label_values

def inputs_unbiased(batch_size, num_epochs,file_name,scaling=True):
    ##This function creates the inputs for training
    filename = file_name

    with tf.name_scope('input_data'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

        pixels, labels = read_and_decode(filename_queue,
                                         scaling=scaling)

        inputs = [pixels,labels]

        pixels_shuffled,labels_shuffled=tf.train.shuffle_batch(inputs,
                                                batch_size,
                                                batch_size * 100,
                                                batch_size * 3,
                                                num_threads=8
                                                               )


        return pixels_shuffled, labels_shuffled



def inputs_biased(batch_size, num_epochs,file_name):
    ##This function creates the inputs for training
    filename = file_name

    with tf.name_scope('input_data'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

        pixels, labels = read_and_decode(filename_queue,scaling=False)
        inputs = [pixels,labels]
        ###Add a shuffling queue to make stochastic
        dtypes = list(map(lambda x: x.dtype, inputs))
        shapes = list(map(lambda x: x.get_shape(), inputs))

       #     queue = tf.RandomShuffleQueue(batch_size * 5,batch_size*1, dtypes)
       #     enqueue_op = queue.enqueue(inputs)
       #     qr = tf.train.QueueRunner(queue, [enqueue_op] * 2)
       #     tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
       #     inputs = queue.dequeue()
       #     for tensor, shape in zip(inputs, shapes):
       #         tensor.set_shape(shape)


        pixels_shuffled, labels_shuffled = tf.contrib.training.stratified_sample(
            [inputs[0]],
            inputs[1],
            [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
            batch_size,
            queue_capacity=batch_size*200,
            threads_per_queue=16,
            init_probs=None)
        return pixels_shuffled, labels_shuffled

def test_inputs(batch_size,file_name):
    ##This function creates the inputs for testing
    filename_test = file_name

    with tf.name_scope('input_test_data'):
        filename_queue = tf.train.string_input_producer([filename_test])

        pixels,labels= read_and_decode(filename_queue,scaling=True)

        inputs = [pixels,labels]

        pixels_shuffled,labels_shuffled=tf.train.shuffle_batch(inputs,
                                                batch_size,
                                                batch_size * 10,
                                                batch_size * 3,
                                                num_threads=8
                                                               )



        return pixels_shuffled,labels_shuffled
