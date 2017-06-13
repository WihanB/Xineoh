import tensorflow as tf
import numpy as np
import os
import time
import myCNN
import queue_maker
import tempfile
from matplotlib import pyplot as plt

logpath = os.path.dirname(os.path.abspath(__file__)) + '/logs/runlog' + time.asctime() + '44'
if not os.path.exists(logpath):
    os.makedirs(logpath)

##DEFINE FILE NAMES
file_name = 'TF_mnist_train_ub'
filename_test = 'TF_mnist_test'



##construct the main graoh

def run_training(Training):
    with tf.Graph().as_default():
        #This value is low for 2 reasons 1) if you look at the math behind ADAM it scales the LR by 10
        #2) Because we have many similar examples for the imbalanced classes overfitting is a threat
        learning_rate=0.001
        # inputs
        sess = tf.Session()
        if not Training:
            num_epochs=0.1
        else:
            num_epochs=10
        ### PHYSICAL GRAPH CONSTRUCTION
        ####
        train_x, train_t = queue_maker.inputs_unbiased(100, num_epochs, file_name, scaling=True)
        test_x, test_t = queue_maker.test_inputs(100, filename_test)

        train_logits = myCNN.cnn_model_fn(train_x, True)
        test_logits = myCNN.cnn_model_fn(test_x, False)

        train_loss, train_summary, train_predictions = myCNN.loss_train(
            train_logits,
            train_t)
        test_loss, test_summary, test_predictions = myCNN.loss_test(
            test_logits,
            test_t)
        ##
        train, save_vars = myCNN.training(train_loss, learning_rate)

        saver=tf.train.Saver(save_vars)

        ######
        ##Initialisation

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        summary_writer = tf.summary.FileWriter(logpath, sess.graph)

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            if not Training:
                saver.restore(sess,"./my_weights")
            else:
                time.sleep(15) #Allows queue prefetching, this helps for GPU


            while not coord.should_stop():
                start_time = time.time()
                if not Training:
                    summary_str, loss_value = sess.run([ train_summary,
                                                           train_loss])
                else:
                    _, summary_str, loss_value = sess.run([train,
                                                           train_summary,
                                                           train_loss])
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                duration = time.time() - start_time
                step += 1
                if step % 5 == 0:
                    pred, summary_str, loss_value = sess.run([test_predictions,
                                                              test_summary,
                                                              test_loss])
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                               duration))
                    print('Step %d: accuracy = %.2f (%.3f sec)' % (step, pred['accuracy'],
                                                                   duration))
                if step % 10000==0:
                    saver.save(sess,'my_weights')
                    saver.restore(sess,'my_weights')


                if step % 2000 == 0:
                    acc = 0
                    for i in range(100):
                        start_time = time.time()
                        pred, summary_str, loss_value = sess.run([test_predictions,
                                                                  test_summary,
                                                                  test_loss])
                        acc += pred['accuracy'] / 100

                    acc = acc / 100.0
                    duration = time.time() - start_time
                    print('Step %d: Full_accuracy = %.3f (%.3f sec)' % (step, acc,
                                                                        duration))


        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (num_epochs, step))
            saver.save(sess, './my_weights')
            saver.restore(sess, './my_weights')
            print("Evaluating Final Accuracy")
            acc = 0
            confusion_mat=np.zeros([10, 10])
            for i in range(100):
                start_time = time.time()
                pred, summary_str, loss_value = sess.run([test_predictions,
                                                          test_summary,
                                                          test_loss])
                acc += pred['accuracy'] / 100
                #print(pred['confusion_matrix'])
                confusion_mat += pred['confusion_matrix']
            acc = acc / 100.0
            duration = time.time() - start_time
            print('Step %d: Full_accuracy = %.3f (%.3f sec)' % (step, acc,
                                                                duration))
            print("Creating Confusion_Matrix")
            np.savetxt('confusion_matrix', confusion_mat, fmt="%d")
        except KeyboardInterrupt:
            print('Closing Threads!')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


def main():
    run_training(Training=True)
    print('Training Complete; execute the following code to view the training process')
    print('python -m tensorflow.tensorboard --logdir /logs')


if __name__ == '__main__':
    main()
