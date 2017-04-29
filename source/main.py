import writer
import reader
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import time
import math
import model




#Todo

class Config():
    #data path
    data_path = '/data/lwq/image-data'

    batch_size = 32
    max_step = 30000
    
    decay = 0.95
    decay_step = 200
    starter_learning_rate = 1e-3

    #variable save and restore
    steps = max_step
    checkpoint_iter = 2000
    params_dir = '../params/'
    save_filename = 'model'
    load_filename = 'model-' + str(steps)

    #summary
    log_dir = '../log/'
    summary_iter = 2000


def init_kernelRes(sess):
    kernelRes = sess.run(tf.truncated_normal([5, 5, 1, 12], dtype=tf.float32, stddev=1e-1))
    k_sum = sess.run(tf.reduce_sum(kernelRes, reduction_indices=[0, 1, 2]))

    for k in range(12):
        for i in range(5):
            for j in range(5):
                if i != 2 or j != 2:
                    kernelRes[i, j, 0, k] /= (k_sum[k]-kernelRes[2, 2, 0, k])
        kernelRes[2, 2, 0, k] = -1

    return kernelRes

def main():
    config = Config()

    #Todo
    #writer.create_record( config.data_path)

    images_train, labels_train = reader.train_inputs(data_dir="../train.tfrecords", batch_size=config.batch_size)

    images_test, labels_test = reader.test_inputs(data_dir="../test.tfrecords", batch_size=config.batch_size)


    modeler = model.Model(config)
    
    logits, _ = modeler.inference()    
    loss = modeler.loss(logits)
    train_op = modeler.train_op(loss)

    top_k = modeler.cal_accuracy(logits)

    #init = tf.global_variables_initializer()
    init = tf.initialize_all_variables()

    saver = tf.train.Saver(max_to_keep=100)


    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
       
        kernelRes_train = init_kernelRes(sess=sess)

        
        merged = tf.summary.merge_all()
        logdir = os.path.join(config.log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        train_writer = tf.summary.File_Writer(logdir, sess.graph)
        

        #train
        print "Start training"
        for step in range(config.max_step):
            start_time = time.time()

            with tf.device("/cpu:0"):
                image_batch, label_batch = sess.run([images_train, labels_train])
                
            feed_dict = {
                modeler.image_holder:image_batch,
                modeler.label_holder:label_batch,
                modeler.kernelRes:kernelRes_train,
                modeler.keep_prob:0.5
            }
        
            with tf.device("/gpu:2"):
                _, loss_value = sess.run([train_op, loss],
                                            feed_dict=feed_dict)

            duration = time.time()-start_time

            if step % 10 == 0:
                examples_per_sec = config.batch_size/duration
                sec_per_batch = float(duration)

                format_str = ('step %d, loss =  %.2f (%.1f examples/sec; %.3f sec/batch)')
                print format_str % (step, loss_value, examples_per_sec, sec_per_batch)

            with tf.device("/cpu:0"):
                #save checkpoint
                if (step+1)%config.checkpoint_iter == 0:
                    saver.save(sess, 
                                config.params_dir+config.save_filename, 
                                modeler.global_step.eval())
                #write summary
                if (step+1)%config.summary_iter == 0:
                    summary = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary, modeler.global_step.eval())
            

        #test
        num_examples = reader.NUM_EXAMPLES_PER_EPOCH_FOR_TEST
        num_iter = int(math.ceil(num_examples/config.batch_size))
        true_count = 0
        total_sample_count = num_iter*config.batch_size
        accuracy = np.zeros(12)
        
        step = 0
        while step < num_iter:
            step += 1
            print step

            with tf.device("/cpu:0"):
                image_batch, label_batch = sess.run([images_test, labels_test])

            with tf.device("/gpu:2"):
                predictions = sess.run([top_k], 
                                    feed_dict={modeler.image_holder:image_batch, 
                                    modeler.label_holder:label_batch, 
                                    modeler.kernelRes:kernelRes_train,
                                    modeler.keep_prob:1.0})
            
            for i in range(config.batch_size):
                if predictions[0][i]:
                    accuracy[label_batch[i]] += 1;

            true_count += np.sum(predictions)

            print true_count
            
            precision = 1.0*true_count / total_sample_count

            print 'precision @ 1 = %.3f' % precision
            print accuracy
        
        accuracy = accuracy*12.0/total_sample_count
        print accuracy

        coord.request_stop()
        coord.join()

if __name__ == "__main__":
    main()
