# -*- coding: UTF-8 -*-
import tensorflow as tf

def vvgg(x2, keep_prob, num_classes):

    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(x2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv11 = tf.nn.relu(bias, name=scope)
        print 'conv11:'
        print conv11.shape

    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3,64, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv11, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv12 = tf.nn.relu(bias, name=scope)
        print 'conv12:'
        print conv12.shape


    with tf.name_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv12,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
        print 'pool1:'
        print pool1.shape
                             

    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv21 = tf.nn.relu(bias, name=scope)
        print 'conv21:'
        print conv21.shape


    with tf.name_scope('conv2_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3,128, 128], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv21, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv22 = tf.nn.relu(bias, name=scope)
        print 'conv22:'
        print conv22.shape


    with tf.name_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv22,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
        print 'pool2:'
        print pool2.shape

    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv31 = tf.nn.relu(bias, name=scope)
        print 'conv31:'
        print conv31.shape


    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3,256, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv31, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv32 = tf.nn.relu(bias, name=scope)
        print 'conv32:'
        print conv32.shape        

    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3,256, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv32, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv33 = tf.nn.relu(bias, name=scope)
        print 'conv33:'
        print conv33.shape

    with tf.name_scope('pool3') as scope:
        pool3 = tf.nn.max_pool(conv33,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
        print 'pool3:'
        print pool3.shape

    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv41 = tf.nn.relu(bias, name=scope)
        print 'conv41:'
        print conv41.shape


    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3,512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv41, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv42 = tf.nn.relu(bias, name=scope)
        print 'conv42:'
        print conv42.shape        

    with tf.name_scope('conv4_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3,512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv42, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv43 = tf.nn.relu(bias, name=scope)
        print 'conv43:'
        print conv43.shape

    with tf.name_scope('pool4') as scope:
        pool4 = tf.nn.max_pool(conv43,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME') 
        print 'pool4:'
        print pool4.shape

    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv51 = tf.nn.relu(bias, name=scope)
        print 'conv51:'
        print conv51.shape


    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3,512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv51, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv52 = tf.nn.relu(bias, name=scope)
        print 'conv52:'
        print conv52.shape

    with tf.name_scope('conv5_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3,512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv52, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv53 = tf.nn.relu(bias, name=scope)
        print 'conv53:'
        print conv53.shape

    with tf.name_scope('pool5') as scope:
        pool5 = tf.nn.max_pool(conv53,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')                                
        print 'pool5:'
        print pool5.shape

    with tf.name_scope('flattened6') as scope:
        flattened = tf.reshape(pool5, shape=[-1, 7*7*512])
        print 'flat.shape'
        print flattened.shape


    with tf.name_scope('fc6') as scope:
        weights = tf.Variable(tf.truncated_normal([7*7*512, 4096],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                            trainable=True, name='biases')
        bias = tf.nn.xw_plus_b(flattened, weights, biases)
        fc6 = tf.nn.relu(bias)
        print 'fc6:shape:'
        print fc6.shape


    with tf.name_scope('fc7') as scope:
        weights = tf.Variable(tf.truncated_normal([4096,4096],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                            trainable=True, name='biases')
        bias = tf.nn.xw_plus_b(fc6, weights, biases)
        fc7 = tf.nn.relu(bias)
        print 'fc7:shape:'
        print fc7.shape

    with tf.name_scope('fc8') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, num_classes],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32),
                                        trainable=True, name='biases')
        fc8 = tf.nn.xw_plus_b(fc7, weights, biases)
        print 'fc8:shape:'
        print fc8.shape
    return fc8