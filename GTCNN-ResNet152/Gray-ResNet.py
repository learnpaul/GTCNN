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
    
    kernel = tf.Variable(tf.truncated_normal([k_h, k_w, c_i, c_o], dtype=tf.float32,
                                         stddev=1e-1), name='weights')                                     
    conv = tf.nn.conv2d(input, kernel, [1, s_h, s_w, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[c_o], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)

    print name+':'
    print bias.shape          
    return bias

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

def residual_block(x,out_channel_1,out_channel_2, strides,downsample, name="unit"):
    in_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tresidual unit::::: %s' % scope.name)
        if downsample:
            x=tf.nn.max_pool(x, [1,2,2, 1], [1, 2, 2, 1], 'VALID')

        if in_channel == out_channel_2:
            if strides == 1:
                shortcut = tf.identity(x)
            else:
                shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
        else:
            shortcut = conv(x, 1,1, out_channel_2, strides,strides, name='shortcut')
        print 'shortcut:::::::::::'
        print shortcut.shape

        x=conv(x,1,1,out_channel_1,1,1, name='con1::::::::::::::')
        x=lrn(x,2, 2e-05, 0.75, name='lrn11')
        x=tf.nn.relu(x, name=name)

        x=conv(x,3,3,out_channel_1,1,1, name='con2::::::::::::::')
        x=lrn(x,2, 2e-05, 0.75, name='lrn12')
        x=tf.nn.relu(x, name=name)

        x=conv(x,1,1,out_channel_2,1,1, name='con3:::::::::::::::::::::::::::::::::::::::::::::::::')
        x=lrn(x,2, 2e-05, 0.75, name='lrn13')
        x=tf.nn.relu(x, name=name)        

        x = x + shortcut
        x =tf.nn.relu(x, name='relu_2')
        print name + '::::::::::::::::::::::::::::'
        print x.shape
    return x


def resnet(x1, keep_prob, num_classes):

    with tf.name_scope('conv1') as scope:
        conv1=conv(x1,7,7,64,2,2, name='conv1')
        lrn1=lrn(conv1,2,2e-05,0.75,name='lrn1')
        relu1=tf.nn.relu(lrn1)
        pool1=max_pool(relu1,3, 3, 2, 2, name='pool1')


    with tf.name_scope('residual_2x') as scope:
        conv2x_1=residual_block(pool1,64,256,1,downsample=False,name='2x_1')
        conv2x_2=residual_block(conv2x_1,64,256,1,downsample=False,name='2x_2')
        conv2x_3=residual_block(conv2x_2,64,256,1,downsample=False,name='2x_3')



    with tf.name_scope('residual_3x') as scope:
        conv3x_1=residual_block(conv2x_3,128,512,1,downsample=True,name='3x_1')
        conv3x_2=residual_block(conv3x_1,128,512,1,downsample=False,name='3x_2')
        conv3x_3=residual_block(conv3x_2,128,512,1,downsample=False,name='3x_3')
        conv3x_4=residual_block(conv3x_3,128,512,1,downsample=False,name='3x_4')
        conv3x_5=residual_block(conv3x_4,128,512,1,downsample=False,name='3x_5')
        conv3x_6=residual_block(conv3x_5,128,512,1,downsample=False,name='3x_6')
        conv3x_7=residual_block(conv3x_6,128,512,1,downsample=False,name='3x_7')
        conv3x_8=residual_block(conv3x_7,128,512,1,downsample=False,name='3x_8')
        

    with tf.name_scope('residual_4x') as scope:
        conv4x_1=residual_block(conv3x_8,256,1024,1,downsample=True,name='4x_1')
        conv4x_2=residual_block(conv4x_1,256,1024,1,downsample=False,name='4x_2')
        conv4x_3=residual_block(conv4x_2,256,1024,1,downsample=False,name='4x_3') 
        conv4x_4=residual_block(conv4x_3,256,1024,1,downsample=False,name='4x_4')
        conv4x_5=residual_block(conv4x_4,256,1024,1,downsample=False,name='4x_5')
        conv4x_6=residual_block(conv4x_5,256,1024,1,downsample=False,name='4x_6') 
        conv4x_7=residual_block(conv4x_6,256,1024,1,downsample=False,name='4x_7')
        conv4x_8=residual_block(conv4x_7,256,1024,1,downsample=False,name='4x_8')
        conv4x_9=residual_block(conv4x_8,256,1024,1,downsample=False,name='4x_9') 
        conv4x_10=residual_block(conv4x_9,256,1024,1,downsample=False,name='4x_10')
        conv4x_11=residual_block(conv4x_10,256,1024,1,downsample=False,name='4x_11')
        conv4x_12=residual_block(conv4x_11,256,1024,1,downsample=False,name='4x_12') 
        conv4x_13=residual_block(conv4x_12,256,1024,1,downsample=False,name='4x_13')
        conv4x_14=residual_block(conv4x_13,256,1024,1,downsample=False,name='4x_14')
        conv4x_15=residual_block(conv4x_14,256,1024,1,downsample=False,name='4x_15')        
        conv4x_16=residual_block(conv4x_15,256,1024,1,downsample=False,name='4x_16')
        conv4x_17=residual_block(conv4x_16,256,1024,1,downsample=False,name='4x_17')
        conv4x_18=residual_block(conv4x_17,256,1024,1,downsample=False,name='4x_18') 
        conv4x_19=residual_block(conv4x_18,256,1024,1,downsample=False,name='4x_19')
        conv4x_20=residual_block(conv4x_19,256,1024,1,downsample=False,name='4x_20')
        conv4x_21=residual_block(conv4x_20,256,1024,1,downsample=False,name='4x_21') 
        conv4x_22=residual_block(conv4x_21,256,1024,1,downsample=False,name='4x_22')
        conv4x_23=residual_block(conv4x_22,256,1024,1,downsample=False,name='4x_23')
        conv4x_24=residual_block(conv4x_23,256,1024,1,downsample=False,name='4x_24') 
        conv4x_25=residual_block(conv4x_24,256,1024,1,downsample=False,name='4x_25')
        conv4x_26=residual_block(conv4x_25,256,1024,1,downsample=False,name='4x_26')
        conv4x_27=residual_block(conv4x_26,256,1024,1,downsample=False,name='4x_27') 
        conv4x_28=residual_block(conv4x_27,256,1024,1,downsample=False,name='4x_28')
        conv4x_29=residual_block(conv4x_28,256,1024,1,downsample=False,name='4x_29')
        conv4x_30=residual_block(conv4x_29,256,1024,1,downsample=False,name='4x_30')
        conv4x_31=residual_block(conv4x_30,256,1024,1,downsample=False,name='4x_31') 
        conv4x_32=residual_block(conv4x_31,256,1024,1,downsample=False,name='4x_32')
        conv4x_33=residual_block(conv4x_32,256,1024,1,downsample=False,name='4x_33')
        conv4x_34=residual_block(conv4x_33,256,1024,1,downsample=False,name='4x_34') 
        conv4x_35=residual_block(conv4x_34,256,1024,1,downsample=False,name='4x_35')
        conv4x_36=residual_block(conv4x_35,256,1024,1,downsample=False,name='4x_36')        

    with tf.name_scope('residual_5x') as scope:
        conv5x_1=residual_block(conv4x_36,512,2048,1,downsample=True,name='5x_1')
        conv5x_2=residual_block(conv5x_1,512,2048,1,downsample=False,name='5x_2')
        conv5x_3=residual_block(conv5x_2,512,2048,1,downsample=False,name='5x_3')


    pool=avg_pool(conv5x_3,7, 7, 7, 7, name='pool_avg')
        

    with tf.name_scope('flattened') as scope:
        flattened = tf.reshape(pool, shape=[-1, 1*1*2048])
        print ':::::::::::flattened:'
        print flattened.shape        

    with tf.name_scope('fc') as scope:
        weights = tf.Variable(tf.truncated_normal([1*1*2048, num_classes],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32),
                            trainable=True, name='biases')
        fc = tf.nn.xw_plus_b(flattened, weights, biases)
        print ':::::::::::fc'
        print fc.shape
    return fc     