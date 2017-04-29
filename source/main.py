import writer
import reader
import os
import tensorflow as tf
import numpy as np
import time
import math
import model

cwd = os.getcwd()
dir = os.path.dirname(cwd)


#Todo
#writer.create_record("train")
#writer.create_record("test")

#Todo

class Config():
    batch_size = 32
    max_step = 10000
    
    decay = 0.95
    decay_step = 200
    starter_learning_rate = 1e-3

    #variable save and restore
    steps = max_step
    checkpoint_iter = 2000
    params_dir = '../params/'
    save_filename = 'model'
    load_filename = 'model-' + steps

    #summary
    log_dir = '../log/'
    summary_iter = 2000

'''
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

def print_activations(tensor):
    print tensor.op.name, ' ', tensor.get_shape().as_list()

def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

def inference(images, kernelRes, keep_prob):
    parameters = []

    with tf.name_scope('convRes') as scope:
        conv = tf.nn.conv2d(images, kernelRes, [1, 1, 1, 1], padding='VALID')
        biasesRes = tf.Variable(tf.constant(0.0, shape=[12], dtype=tf.float32), trainable=True, name='biases')

        print_activations(conv)
        print_activations(biasesRes)
        parameters += [biasesRes]
    
    convRes = tf.nn.bias_add(conv, biasesRes, name=scope)
    print_activations(convRes)
    
    #conv1
    with tf.name_scope('conv1') as scope:
        kernel = variable_with_weight_loss(shape=[7, 7, 12, 64], stddev=5e-2, wl=0.0)
        conv = tf.nn.conv2d(convRes, kernel, [1, 2, 2, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]

        print_activations(conv1)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    lrn1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')

    print_activations(pool1)
    print_activations(lrn1)

    #conv2
    with tf.name_scope('conv2') as scope:
        kernel = variable_with_weight_loss(shape=[5, 5, 64, 48], stddev=5e-2, wl=0.0)
        conv = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[48], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]

        print_activations(conv2)
    
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    lrn2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
    
    print_activations(pool2)
    print_activations(lrn2)

    #fc1
    with tf.name_scope('fc1') as scope:
        reshape = tf.reshape(lrn2, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_loss(shape=[dim, 4096], stddev=0.04, wl=0.0)
        #biases = tf.Variable(tf.constant(0.1, shape='4096', dtype=tf.float32), )
        biases = tf.Variable(tf.zeros([4096]), name='biases')
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope)

        parameters += [weights, biases]
        print_activations(fc1)

    drop1 = tf.nn.dropout(fc1, keep_prob, name='drop1')
    print_activations(drop1)

    #fc2
    with tf.name_scope('fc2') as scope:
        weights = variable_with_weight_loss(shape=[4096, 4096], stddev=0.04, wl=4)

        biases = tf.Variable(tf.zeros([4096]), name='biases')
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope)

        parameters += [weights, biases]
        print_activations(fc2)

    drop2 = tf.nn.dropout(fc2, keep_prob, name='drop2')
    print_activations(drop2)

    #fc3
    with tf.name_scope('fc3') as scope:
        weights = variable_with_weight_loss(shape=[4096, 5], stddev=1/4096.0, wl=4)
        biases = tf.Variable(tf.zeros([5]), name='biases')
    logits = tf.add(tf.matmul(drop2, weights), biases)
    #logits = tf.nn.softmax(tf.matmul(drop2, weights) + biases, name='logits')
    print_activations(logits)

    return logits, parameters

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_perexample')

    cross_entropy = tf.reduce_mean(cross_entropy_sum, name='cross_entropy')

    tf.add_to_collection('losses', cross_entropy)

    return tf.add_n(tf.get_collection('losses'), name='total_loss') 

def train_op(total_loss):
    
    global_step = tf.get_variable('global_step', initializer=0, dtype=tf.int32, trainable=False)
    
    """
    optimizer = tf.train.AdamOptimizer(starter_learning_rate)
    grads = optimizer.compute_gradients(total_loss)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(decay, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')
    """
    
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_step, decay, staircase=True)
    # Passing global_step to minimize() will increment it at each step.
    train_op = (
            tf.train.AdamOptimizer(learning_rate)
            .minimize(total_loss, global_step=global_step)
    )
    

    return train_op
'''

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

    images_train, labels_train = reader.distorted_inputs(data_dir=dir+"/train.tfrecords", batch_size=config.batch_size)

    images_test, labels_test = reader.inputs(data_dir=dir+"/test.tfrecords", batch_size=config.batch_size)



    modeler = model.Model(config)
    
    logits, _ = modeler.inference()    
    loss = modeler.loss(logits)
    train_op = modeler.train_op(loss)

    top_k = modeler.cal_accuracy(logits)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=100)


    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
       
        kernelRes_train = init_kernelRes(sess=sess)

        #train
        print "Start training"
        for step in range(max_step):
            start_time = time.time()

            with tf.device("/cpu:0"):
                image_batch, label_batch = sess.run([images_train, labels_train])
                
            with tf.device("/gpu:2"):
                _, loss_value = sess.run([train_op, loss],
                                            feed_dict={modeler.image_holder:image_batch,
                                            modeler.label_holder:label_batch,
                                            modeler.kernelRes:kernelRes_train,
                                            modeler.keep_prob:0.5})

            duration = time.time()-start_time

            if step % 10 == 0:
                examples_per_sec = config.batch_size/duration
                sec_per_batch = float(duration)

                format_str = ('step %d, loss =  %.2f (%.1f examples/sec; %.3f sec/batch)')
                print format_str % (step, loss_value, examples_per_sec, sec_per_batch)

                #print kernelRes_train
            
            if (step+1)%config.checkpoint_iter == 0:
                saver.save(sess, 
                        config.params_dir+config.save_filename, 
                        modeler.global_step.eval())


        #test
        num_examples = reader.NUM_EXAMPLES_PER_EPOCH_FOR_TEST
        num_iter = int(math.ceil(num_examples/config.batch_size))
        true_count = 0
        total_sample_count = num_iter*config.batch_size
        accuracy = np.zeros(5)
        
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
        
        accuracy = accuracy*5.0/total_sample_count
        print accuracy

        coord.request_stop()
        coord.join()
