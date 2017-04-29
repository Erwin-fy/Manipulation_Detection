#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf


class Model():
    def __init__(self, config):
        self.global_step = tf.get_variable('global_step', initializer=0,
                            dtype=f.int32, trainable=False)

        self.batch_size = config.batch_size
        self.decay = config.decay
        self.decay_step = config.decay_step
        self.starter_learning_rate = config.starter_learning_rate
        
        self.image_holder = tf.placeholder(tf.float32, [self.batch_size, 227, 227, 1])
        self.label_holder = tf.placeholder(tf.int32, [self.batch_size])
        self.keep_prob = tf.placeholder(tf.float32)
        self.kernelRes = tf.placeholder(tf.float32, [5, 5, 1, 12])


    def print_activations(self, tensor):
        print tensor.op.name, ' ', tensor.get_shape().as_list()

    def variable_with_weight_loss(self, shape, stddev, wl):
        var = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=stddev))
        if wl is not None:
            weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
            tf.add_to_collection('losses', weight_loss)
        return var

    def inference(self):
        parameters = []

        with tf.name_scope('convRes') as scope:
            conv = tf.nn.conv2d(self.images, self.kernelRes, [1, 1, 1, 1], padding='VALID')
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

    def loss(self, logits):
        labels = tf.cast(self.label_holder, tf.int64)
        cross_entropy_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=logits, labels=labels, name='cross_entropy_perexample')

        cross_entropy = tf.reduce_mean(cross_entropy_sum, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy)

        loss_value = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar('loss', loss_value)

        return loss_value
    
    def train_op(self, total_loss):
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 
                            self.decay_step, self.decay, staircase=True)
        train_op = (
                tf.train.AdamOptimizer(learning_rate)
                .minimize(total_loss, global_step=self.global_step)
        )
        

        return train_op
        
    def cal_accuracy(self, logits):
        return tf.nn.in_top_k(logits, self.label_holder, 1)

    def activation_summary(self, activation):
        name = activation.op.name
        tf.summary.histogram(name + '/activations', activation)
        tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(activation))

    def logits_summary(self, logits):
        pass