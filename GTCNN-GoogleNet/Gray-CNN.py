# -*- coding: UTF-8 -*-
import tensorflow as tf

def conv(input,
         k_h,
         k_w,
         c_o,
         s_h,
         s_w,
         name
        ):
 
    c_i=input.get_shape().as_list()[-1]
    print c_i
    print type(c_i)
    
    kernel = tf.Variable(tf.truncated_normal([k_h, k_w, c_i, c_o], dtype=tf.float32,
                                         stddev=1e-1), name='weights')
    print kernel                                     
    conv = tf.nn.conv2d(input, kernel, [1, s_h, s_w, 1], padding='SAME')
    print conv
    biases = tf.Variable(tf.constant(0.0, shape=[c_o], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    convo = tf.nn.relu(bias, name=name)
    print name+':'
    print convo.shape          
    return convo

def max_pool(input, k_h, k_w, s_h, s_w, name):
  
    pool = tf.nn.max_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding='SAME',
                          name=name)
    print name+':'
    print pool.shape    
    return pool

def avg_pool(input, k_h, k_w, s_h, s_w, name):
    
    pool_av=tf.nn.avg_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding='SAME',
                          name=name)
    print name+':'
    print pool_av.shape    
    return pool_av                            

def lrn(input, radius, alpha, beta, name, bias=1.0):
    lrn=tf.nn.local_response_normalization(input=input,
                                          depth_radius=radius,
                                          alpha=alpha,
                                          beta=beta,
                                          bias=bias,
                                          name=name)
    return lrn


def inception_3a(inputs, scope=None, reuse=None):
    with tf.name_scope('Branch_0'):
        branch_0 = conv(inputs,1,1,64,1,1,name='3a_branch0')
    with tf.name_scope('Branch_1'):
        branch_1 = conv(inputs,1,1,96,1,1,name='3a_branch10')
        branch_1 = conv(branch_1,3,3,128,1,1,name='3a_branch11')
    with tf.name_scope('Branch_2'):
        branch_2 = conv(inputs,1,1,16,1,1,name='3a_branch20')
        branch_2 = conv(branch_2,5,5,32,1,1,name='3a_branch21')
    with tf.name_scope('Branch_3'):
        branch_3 = max_pool(inputs,3,3,1,1,name='3a_branch30')
        branch_3 = conv(branch_3,1,1,32,1,1,name='3a_branch31')
    concat=tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3]) 
    print '::::::::::::::inception_3a:'
    print concat.shape    
    return concat


def inception_3b(inputs, scope=None, reuse=None):
    with tf.name_scope('Branch_0'):
        branch_0 = conv(inputs,1,1,128,1,1,name='3a_branch0')
    with tf.name_scope('Branch_1'):
        branch_1 = conv(inputs,1,1,128,1,1,name='3a_branch10')
        branch_1 = conv(branch_1,3,3,192,1,1,name='3a_branch11')
    with tf.name_scope('Branch_2'):
        branch_2 = conv(inputs,1,1,32,1,1,name='3a_branch20')
        branch_2 = conv(branch_2,5,5,96,1,1,name='3a_branch21')
    with tf.name_scope('Branch_3'):
        branch_3 = max_pool(inputs,3,3,1,1,name='3a_branch30')
        branch_3 = conv(branch_3,1,1,64,1,1,name='3a_branch31')
    concat=tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3]) 
    print '::::::::::::::inception_3b:'
    print concat.shape    
    return concat    

def inception_4a(inputs, scope=None, reuse=None):
    with tf.name_scope('Branch_0'):
        branch_0 = conv(inputs,1,1,192,1,1,name='3a_branch0')
    with tf.name_scope('Branch_1'):
        branch_1 = conv(inputs,1,1,96,1,1,name='3a_branch10')
        branch_1 = conv(branch_1,3,3,208,1,1,name='3a_branch11')
    with tf.name_scope('Branch_2'):
        branch_2 = conv(inputs,1,1,16,1,1,name='3a_branch20')
        branch_2 = conv(branch_2,5,5,48,1,1,name='3a_branch21')
    with tf.name_scope('Branch_3'):
        branch_3 = max_pool(inputs,3,3,1,1,name='3a_branch30')
        branch_3 = conv(branch_3,1,1,64,1,1,name='3a_branch31')
    concat=tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3]) 
    print '::::::::::::::inception_4a:'
    print concat.shape        
    return concat    

def inception_4b(inputs, scope=None, reuse=None):
    with tf.name_scope('Branch_0'):
        branch_0 = conv(inputs,1,1,160,1,1,name='3a_branch0')
    with tf.name_scope('Branch_1'):
        branch_1 = conv(inputs,1,1,112,1,1,name='3a_branch10')
        branch_1 = conv(branch_1,3,3,224,1,1,name='3a_branch11')
    with tf.name_scope('Branch_2'):
        branch_2 = conv(inputs,1,1,24,1,1,name='3a_branch20')
        branch_2 = conv(branch_2,5,5,64,1,1,name='3a_branch21')
    with tf.name_scope('Branch_3'):
        branch_3 = max_pool(inputs,3,3,1,1,name='3a_branch30')
        branch_3 = conv(branch_3,1,1,64,1,1,name='3a_branch31')
    concat=tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    print '::::::::::::::inception_4b:'
    print concat.shape        
    return concat    
    
def inception_4c(inputs, scope=None, reuse=None):
    with tf.name_scope('Branch_0'):
        branch_0 = conv(inputs,1,1,128,1,1,name='3a_branch0')
    with tf.name_scope('Branch_1'):
        branch_1 = conv(inputs,1,1,128,1,1,name='3a_branch10')
        branch_1 = conv(branch_1,3,3,256,1,1,name='3a_branch11')
    with tf.name_scope('Branch_2'):
        branch_2 = conv(inputs,1,1,24,1,1,name='3a_branch20')
        branch_2 = conv(branch_2,5,5,64,1,1,name='3a_branch21')
    with tf.name_scope('Branch_3'):
        branch_3 = max_pool(inputs,3,3,1,1,name='3a_branch30')
        branch_3 = conv(branch_3,1,1,64,1,1,name='3a_branch31')
    concat=tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    print '::::::::::::::inception_4c:'
    print concat.shape        
    return concat   

def inception_4d(inputs, scope=None, reuse=None):
    with tf.name_scope('Branch_0'):
        branch_0 = conv(inputs,1,1,112,1,1,name='3a_branch0')
    with tf.name_scope('Branch_1'):
        branch_1 = conv(inputs,1,1,144,1,1,name='3a_branch10')
        branch_1 = conv(branch_1,3,3,288,1,1,name='3a_branch11')
    with tf.name_scope('Branch_2'):
        branch_2 = conv(inputs,1,1,32,1,1,name='3a_branch20')
        branch_2 = conv(branch_2,5,5,64,1,1,name='3a_branch21')
    with tf.name_scope('Branch_3'):
        branch_3 = max_pool(inputs,3,3,1,1,name='3a_branch30')
        branch_3 = conv(branch_3,1,1,64,1,1,name='3a_branch31')
    concat=tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3]) 
    print '::::::::::::::inception_4d:'
    print concat.shape        
    return concat   

def inception_4e(inputs, scope=None, reuse=None):
    with tf.name_scope('Branch_0'):
        branch_0 = conv(inputs,1,1,256,1,1,name='3a_branch0')
    with tf.name_scope('Branch_1'):
        branch_1 = conv(inputs,1,1,160,1,1,name='3a_branch10')
        branch_1 = conv(branch_1,3,3,320,1,1,name='3a_branch11')
    with tf.name_scope('Branch_2'):
        branch_2 = conv(inputs,1,1,32,1,1,name='3a_branch20')
        branch_2 = conv(branch_2,5,5,128,1,1,name='3a_branch21')
    with tf.name_scope('Branch_3'):
        branch_3 = max_pool(inputs,3,3,1,1,name='3a_branch30')
        branch_3 = conv(branch_3,1,1,128,1,1,name='3a_branch31')
    concat=tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    print '::::::::::::::inception_4e:'
    print concat.shape        
    return concat   
    
def inception_5a(inputs, scope=None, reuse=None):
    with tf.name_scope('Branch_0'):
        branch_0 = conv(inputs,1,1,256,1,1,name='3a_branch0')
    with tf.name_scope('Branch_1'):
        branch_1 = conv(inputs,1,1,160,1,1,name='3a_branch10')
        branch_1 = conv(branch_1,3,3,320,1,1,name='3a_branch11')
    with tf.name_scope('Branch_2'):
        branch_2 = conv(inputs,1,1,32,1,1,name='3a_branch20')
        branch_2 = conv(branch_2,5,5,128,1,1,name='3a_branch21')
    with tf.name_scope('Branch_3'):
        branch_3 = max_pool(inputs,3,3,1,1,name='3a_branch30')
        branch_3 = conv(branch_3,1,1,128,1,1,name='3a_branch31')
    concat=tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    print '::::::::::::::inception_5a:'
    print concat.shape        
    return concat       


def inception_5b(inputs, scope=None, reuse=None):
    with tf.name_scope('Branch_0'):
        branch_0 = conv(inputs,1,1,384,1,1,name='3a_branch0')
    with tf.name_scope('Branch_1'):
        branch_1 = conv(inputs,1,1,192,1,1,name='3a_branch10')
        branch_1 = conv(branch_1,3,3,384,1,1,name='3a_branch11')
    with tf.name_scope('Branch_2'):
        branch_2 = conv(inputs,1,1,48,1,1,name='3a_branch20')
        branch_2 = conv(branch_2,5,5,128,1,1,name='3a_branch21')
    with tf.name_scope('Branch_3'):
        branch_3 = max_pool(inputs,3,3,1,1,name='3a_branch30')
        branch_3 = conv(branch_3,1,1,128,1,1,name='3a_branch31')
    concat=tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    print '::::::::::::::inception_5b:'
    print concat.shape        
    return concat   


def googlenet(x1, keep_prob, num_classes):
    #1-----2 
    with tf.name_scope('conv1_s2_7_7') as scope:
        conv1=conv(x1,7,7,64,2,2, name='conv1_s2_7_7')
        
    with tf.name_scope('pool1_s2_3_3') as scope:
        pool1=max_pool(conv1,3, 3, 2, 2, name='pool1_s2_3_3')
       
    with tf.name_scope('pool1_norm') as scope:
        lrn1=lrn(pool1,2, 2e-05, 0.75, name='pool1_norm')
    #3----4
    with tf.name_scope('conv2_reduce_3_3') as scope:
        conv21=conv(lrn1,1,1,64,1,1, name='conv2_reduce_3_3')
        
    with tf.name_scope('conv2_s1_3_3') as scope:
        conv2=conv(conv21,3,3,192,1,1, name='conv2_s1_3_3')        
       
    with tf.name_scope('conv2_norm') as scope:
        lrn2=lrn(conv2,2, 2e-05, 0.75, name='conv2_norm')
        
    with tf.name_scope('pool2_s2_3_3') as scope:
        pool2=max_pool(lrn2,3, 3, 2, 2, name='pool2_s2_3_3')

    with tf.name_scope('inception_3a') as scope: 
        inception_3_a=inception_3a(pool2)        

    with tf.name_scope('inception_3b') as scope: 
        inception_3_b=inception_3b(inception_3_a)   

    with tf.name_scope('pool3b_4a') as scope:
        pool3=max_pool(inception_3_b,3, 3, 2, 2, name='pool3b_4a')    

    with tf.name_scope('inception_4a') as scope: 
        inception_4_a=inception_4a(pool3)   

    with tf.name_scope('inception_4b') as scope: 
        inception_4_b=inception_4b(inception_4_a)   

    with tf.name_scope('inception_4c') as scope: 
        inception_4_c=inception_4c(inception_4_b)   

    with tf.name_scope('inception_4d') as scope: 
        inception_4_d=inception_4d(inception_4_c)   

    with tf.name_scope('inception_4e') as scope: 
        inception_4_e=inception_4e(inception_4_d)   

    with tf.name_scope('pool4e_5a') as scope:
        pool4=max_pool(inception_4_e,3, 3, 2, 2, name='pool4e_5a') 

    with tf.name_scope('inception_5a') as scope: 
        inception_5_a=inception_5a(pool4)  

    with tf.name_scope('inception_5b') as scope: 
        inception_5_b=inception_5b(inception_5_a)  

    with tf.name_scope('pool_avg') as scope:
        pool=avg_pool(inception_5_b,7, 7, 7, 7, name='pool_avg')

    with tf.name_scope('flattened6') as scope:
        flattened = tf.reshape(pool, shape=[-1, 1*1*1024])
        print ':::::::::::flattened:'
        print flattened.shape        
    with tf.name_scope('dropout') as scope:
       dropout = tf.nn.dropout(flattened, keep_prob)
       print ':::::::::::dropout'
       print dropout.shape
    with tf.name_scope('fc') as scope:
        weights = tf.Variable(tf.truncated_normal([1*1*1024, num_classes],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32),
                            trainable=True, name='biases')
        fc = tf.nn.xw_plus_b(dropout, weights, biases)
        print ':::::::::::fc'
        print fc.shape
    return fc     