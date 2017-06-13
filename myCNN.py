import tensorflow as tf


def cnn_model_fn(features, mode):
    """Model function for CNN."""
    # Input Layer
    if mode!=True:
        reuse=True
    else:
        reuse=False

    input_layer = tf.reshape(features, [-1, 28, 28, 1])
    with tf.name_scope("Convolution"):
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            reuse=reuse,
            name='conv1')
        ##Use batch normalisation as extra regularisation
        conv1_norm=tf.layers.batch_normalization(conv1)
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1_norm, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            reuse=reuse,
            name='conv2')

        conv2_norm=tf.layers.batch_normalization(conv2)

        pool2 = tf.layers.max_pooling2d(inputs=conv2_norm, pool_size=[2, 2], strides=2)

        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    with tf.name_scope("Feedworward"):
        # Dense Layer
        dense_1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu,reuse=reuse,name='FFN1')

        dropout_1 = tf.layers.dropout(
            inputs=dense_1, rate=0.35, training=mode)

        dense_2 = tf.layers.dense(inputs=dropout_1,
                                  units=1024,
                                  activation=tf.nn.relu,
                                  reuse=reuse,
                                  name='FFN2')

        dropout_2 = tf.layers.dropout(
            inputs=dense_2, rate=0.35, training=mode)
        # Logits Layer
        logits = tf.layers.dense(inputs=dropout_2, units=10,reuse=reuse,name='logits')

    return logits

    # Calculate Loss (for both TRAIN and EVAL modes)

def loss_train(logits,labels):
    with tf.name_scope("Train_loss_calculation"):
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

        loss_summary=tf.summary.scalar("Training_loss", loss)

        predictions = {
            "classes": tf.argmax(
                input=logits, axis=1),
            "probabilities": tf.nn.softmax(
                logits, name="softmax_tensor"),
            "accuracy": tf.reduce_sum(tf.cast(
                tf.nn.in_top_k(logits,tf.cast(labels, tf.int32),1),tf.float32))/100.0
        }
    return loss, loss_summary, predictions

def loss_test(logits,labels):
    with tf.name_scope("Test_loss_calculation"):
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

        loss_summary=tf.summary.scalar("Test_loss", loss)

        predictions = {
            "classes": tf.argmax(
                input=logits, axis=1),
            "probabilities": tf.nn.softmax(
                logits, name="softmax_tensor"),
            "accuracy": tf.reduce_sum(tf.cast(
                tf.nn.in_top_k(logits,tf.cast(labels, tf.int32),1),tf.float32)),
            "confusion_matrix": tf.contrib.metrics.confusion_matrix(tf.argmax(
                input=logits, axis=1),labels)
        }

    return loss, loss_summary, predictions

def training(loss,learning_rate,decay=True):

    global_step = tf.get_variable('global_step', initializer=tf.constant(1.0), trainable=False)
    #This performs an simulated annealing esque approach
    if decay:
        learning_rate=tf.train.exponential_decay(learning_rate,
                                                 global_step,
                                                 500, 0.95,staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    ###In order to improve accuracy and kill overfitting, use EMA
    ema=tf.train.ExponentialMovingAverage(decay=0.9999)
    maintain_averages_op=ema.apply(tf.trainable_variables())
    #control dependencies ensures train op runs
    with tf.control_dependencies([train_op]):
        training_op=tf.group(maintain_averages_op)
        restore_vars=ema.variables_to_restore()
    return training_op, restore_vars