# -*- coding: UTF-8 -*-
"""
writen by stephen
"""
"""
modified by chaye
"""
import os
import numpy as np
import tensorflow as tf
from Gray-CNN import alexnet
from Texture-CNN import alexnett
from datagenerator import ImageDataGenerator
from datetime import datetime
import glob
from tensorflow.python.data import Iterator
import pdb


def main():

  
    
    learning_rate = 1e-4
    num_epochs = 30  
    train_batch_size = 100 
    test_batch_size = 100
    dropout_rate = 0.5
    num_classes = 20  
    display_step = 2 

    filewriter_path = "./tmp/tensorboard"  # 存储tensorboard文件
    checkpoint_path = "./tmp/checkpoints"  # 训练好的模型和参数存放目录

    def gray_train():
        image_format = 'png' 
        file_name_of_class = ['acuminatum',
                              'amphiceros',
                              'augur',
                              'bilunaris',
                              'brebissonii',
                              'brevissima',
                              'capitata',
                              'follis',
                              'forcipata',
                              'granulata',
                              'meneghiniana',
                              'mesodon',
                              'oblongella',
                              'potamos',
                              'radiosa',
                              'rhombelliptica',
                              'rhombicum',
                              'sorex',
                              'staurophorum',
                              'venter'                          
                              ] 
        train_dataset_paths = ['/home/hao/Documents/image/two_layer_20s1/train_1/train/acuminatum/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/amphiceros/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/augur/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/bilunaris/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/brebissonii/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/brevissima/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/capitata/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/follis/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/forcipata/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/granulata/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/meneghiniana/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/mesodon/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/oblongella/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/potamos/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/radiosa/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/rhombelliptica/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/rhombicum/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/sorex/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/staurophorum/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train/venter/'
                               ]
        
        train_image_paths = []
        train_labels = []
      
        for train_dataset_path in train_dataset_paths:
            length = len(train_image_paths)
            train_image_paths[length:length] = np.array(glob.glob(train_dataset_path + '*.' + image_format)).tolist()
        for image_path in train_image_paths:
            image_file_name = image_path.split('/')[-2]
            for i in range(num_classes):
                if file_name_of_class[i] in image_file_name:
                    train_labels.append(i)
                    break
       
        
    
        train_data = ImageDataGenerator(
            images=train_image_paths,
            labels=train_labels,
            batch_size=train_batch_size,
            num_classes=num_classes,
            output_buffer_size=10000,
            mode='training',
            image_format=image_format,       
            shuffle=True)
        
        return train_data
    

    def texture_train():
        image_format = 'png' 
        file_name_of_class = ['acuminatum',
                              'amphiceros',
                              'augur',
                              'bilunaris',
                              'brebissonii',
                              'brevissima',
                              'capitata',
                              'follis',
                              'forcipata',
                              'granulata',
                              'meneghiniana',
                              'mesodon',
                              'oblongella',
                              'potamos',
                              'radiosa',
                              'rhombelliptica',
                              'rhombicum',
                              'sorex',
                              'staurophorum',
                              'venter'                          
                              ] 
        train_dataset_paths = ['/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/acuminatum/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/amphiceros/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/augur/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/bilunaris/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/brebissonii/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/brevissima/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/capitata/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/follis/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/forcipata/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/granulata/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/meneghiniana/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/mesodon/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/oblongella/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/potamos/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/radiosa/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/rhombelliptica/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/rhombicum/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/sorex/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/staurophorum/',
                               '/home/hao/Documents/image/two_layer_20s1/train_1/train_gray/venter/'
                               ]


        train_image_paths = []
        train_labels = []

        for train_dataset_path in train_dataset_paths:
            length = len(train_image_paths)
            train_image_paths[length:length] = np.array(glob.glob(train_dataset_path + '*.' + image_format)).tolist()
        for image_path in train_image_paths:
            image_file_name = image_path.split('/')[-2]
            for i in range(num_classes):
                if file_name_of_class[i] in image_file_name:
                    train_labels.append(i)
                    break


        train_data = ImageDataGenerator(
            images=train_image_paths,
            labels=train_labels,
            batch_size=train_batch_size,
            num_classes=num_classes,
            output_buffer_size=10000,
            mode='training',
            image_format=image_format,       
            shuffle=True)


        return train_data

    def test_gray():
        image_format = 'png' 
        file_name_of_class = ['acuminatum',
                              'amphiceros',
                              'augur',
                              'bilunaris',
                              'brebissonii',
                              'brevissima',
                              'capitata',
                              'follis',
                              'forcipata',
                              'granulata',
                              'meneghiniana',
                              'mesodon',
                              'oblongella',
                              'potamos',
                              'radiosa',
                              'rhombelliptica',
                              'rhombicum',
                              'sorex',
                              'staurophorum',
                              'venter'                          
                              ] # cat对应标签0,dog对应标签1。默认图片包含独特的名词，比如类别

        test_dataset_paths =  ['/home/hao/Documents/image/two_layer_20s1/test_1/acuminatum/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/amphiceros/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/augur/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/bilunaris/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/brebissonii/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/brevissima/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/capitata/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/follis/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/forcipata/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/granulata/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/meneghiniana/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/mesodon/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/oblongella/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/potamos/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/radiosa/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/rhombelliptica/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/rhombicum/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/sorex/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/staurophorum/',
                               '/home/hao/Documents/image/two_layer_20s1/test_1/venter/'
                               ]




        test_image_paths = [] 
        test_labels = []

        for test_dataset_path in test_dataset_paths:
            length = len(test_image_paths)		
            test_image_paths[length:length] = np.array(glob.glob(test_dataset_path + '*.' + image_format)).tolist()
        for image_path in test_image_paths:
            image_file_name = image_path.split('/')[-2]
            for i in range(num_classes):
                if file_name_of_class[i] in image_file_name:
                    test_labels.append(i)
                    break        


        test_data = ImageDataGenerator(
            images=test_image_paths,
            labels=test_labels,
            batch_size=test_batch_size,
            num_classes=num_classes,
            output_buffer_size=10000,
            mode='inference',
            image_format=image_format,        
            shuffle=False)       
        return test_data

    def test_texture():
        image_format = 'png' 
        file_name_of_class = ['acuminatum',
                              'amphiceros',
                              'augur',
                              'bilunaris',
                              'brebissonii',
                              'brevissima',
                              'capitata',
                              'follis',
                              'forcipata',
                              'granulata',
                              'meneghiniana',
                              'mesodon',
                              'oblongella',
                              'potamos',
                              'radiosa',
                              'rhombelliptica',
                              'rhombicum',
                              'sorex',
                              'staurophorum',
                              'venter'                          
                              ] 

        test_dataset_paths =  ['/home/hao/Documents/image/two_layer_20s1/test_hist/acuminatum/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/amphiceros/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/augur/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/bilunaris/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/brebissonii/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/brevissima/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/capitata/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/follis/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/forcipata/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/granulata/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/meneghiniana/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/mesodon/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/oblongella/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/potamos/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/radiosa/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/rhombelliptica/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/rhombicum/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/sorex/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/staurophorum/',
                               '/home/hao/Documents/image/two_layer_20s1/test_hist/venter/'
                               ]



        test_image_paths = [] 
        test_labels = []

        for test_dataset_path in test_dataset_paths:
            length = len(test_image_paths)		
            test_image_paths[length:length] = np.array(glob.glob(test_dataset_path + '*.' + image_format)).tolist()
        for image_path in test_image_paths:
            image_file_name = image_path.split('/')[-2]
            for i in range(num_classes):
                if file_name_of_class[i] in image_file_name:
                    test_labels.append(i)
                    break        


        test_data = ImageDataGenerator(
            images=test_image_paths,
            labels=test_labels,
            batch_size=test_batch_size,
            num_classes=num_classes,
            output_buffer_size=10000,
            mode='inference',
            image_format=image_format,        
            shuffle=False)       
        return test_data        
    x1 = tf.placeholder(tf.float32, [train_batch_size, 227, 227, 3])
    y1 = tf.placeholder(tf.float32, [train_batch_size, num_classes])
    print 'x1:'
    print x1.shape
    x2 = tf.placeholder(tf.float32, [train_batch_size, 227, 227, 3])
    print 'x2:'
    print x2.shape    
    y2 = tf.placeholder(tf.float32, [train_batch_size, num_classes])

    x3=tf.placeholder(tf.float32,[100])
    y3=tf.placeholder(tf.float32,[100])
    x4=tf.placeholder(tf.float32,[100])    
    keep_prob = tf.placeholder(tf.float32) 

    train_data1=gray_train()
    test_data1=test_gray()
    with tf.name_scope('input_x1'):

        train_iterator1 = tf.data.Iterator.from_structure(train_data1.data.output_types,
                                        train_data1.data.output_shapes)
        training_initalizer1=train_iterator1.make_initializer(train_data1.data)

        train_next_batch1 = train_iterator1.get_next()

        test_iterator1 = tf.data.Iterator.from_structure(test_data1.data.output_types,
                                        test_data1.data.output_shapes)
        testing_initalizer1=test_iterator1.make_initializer(test_data1.data)

        test_next_batch1 = test_iterator1.get_next()    

    train_data2=texture_train()
    test_data2=test_texture()
    with tf.name_scope('input_x2'):

        train_iterator2 = tf.data.Iterator.from_structure(train_data2.data.output_types,
                                        train_data2.data.output_shapes)
        training_initalizer2=train_iterator2.make_initializer(train_data2.data)
        
        train_next_batch2 = train_iterator2.get_next()
        
        test_iterator2 = tf.data.Iterator.from_structure(test_data2.data.output_types,
                                        test_data2.data.output_shapes)
        testing_initalizer2=test_iterator2.make_initializer(test_data2.data)
       
        test_next_batch2 = test_iterator2.get_next()

    fc8 = alexnet(x1,keep_prob, num_classes)
    fcc8 = alexnett(x2,keep_prob, num_classes)

    with tf.name_scope('loss'):    
        loss_op1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc8,
                                                                  labels=y1))
        loss_op2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fcc8,
                                                                  labels=y2)) 
        print "loss_op1:::::::"
        print loss_op1.shape
        
        print "loss_op2:::::::"
        print loss_op2.shape        
                

    with tf.name_scope('optimizer'):      
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op1 = optimizer.minimize(loss_op1)
        train_op2 = optimizer.minimize(loss_op2)

    with tf.name_scope("accuracy"):
        f1=tf.argmax(fc8,1)
        a1=tf.argmax(y1,1)
        correct_pred1 = tf.equal(f1,a1)
        correct_f1=tf.cast(correct_pred1, tf.float32)
        accuracy1 = tf.reduce_mean(correct_f1)
        
        f2=tf.argmax(fcc8,1)
        a2=tf.argmax(y2,1)
        correct_pred2 = tf.equal(f2,a2)
        correct_f2=tf.cast(correct_pred2, tf.float32)
        

        accuracy2 = tf.reduce_mean(correct_f2)
        correct_f=tf.where(correct_pred1,x=correct_pred1,y=correct_pred2,name=None)        
        correct_f=tf.cast(correct_f,tf.float32)        
        accuracy = tf.reduce_mean(correct_f)         
     
    init = tf.global_variables_initializer()


    tf.summary.scalar('loss1', loss_op1)
    tf.summary.scalar('accuracy1', accuracy1)
    tf.summary.scalar('loss2', loss_op2)
    tf.summary.scalar('accuracy2', accuracy2)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(filewriter_path)


    saver = tf.train.Saver()


    train_batches_per_epoch1 = int(np.floor(train_data1.data_size / train_batch_size))
    test_batches_per_epoch1 = int(np.floor(test_data1.data_size / test_batch_size))



    with tf.Session() as sess:
        sess.run(init)

        writer.add_graph(sess.graph)

        print("{}: Start training...".format(datetime.now()))
        print("{}: Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          filewriter_path))

        for epoch in range(num_epochs):
            sess.run(training_initalizer1)
            sess.run(training_initalizer2)
            print("{}: Epoch number: {} start".format(datetime.now(), epoch + 1))

            
            for step in range(train_batches_per_epoch1):
                img_batch1, label_batch1 = sess.run(train_next_batch1)
                img_batch2, label_batch2 = sess.run(train_next_batch2)                
                loss1,_ = sess.run([loss_op1,train_op1], feed_dict={x1: img_batch1,
                                               y1: label_batch1,
                                               keep_prob: dropout_rate})
                loss2,_ = sess.run([loss_op2,train_op2], feed_dict={x2: img_batch2,
                                               y2: label_batch2,
                                               keep_prob: dropout_rate})                                               
                if step % display_step == 0:
                    # loss
                    print("{}: loss1 = {}".format(datetime.now(), loss1))
                    print '...............................................'
                    print("{}: loss2 = {}".format(datetime.now(), loss2))
                    # Tensorboard
                    s = sess.run(merged_summary, feed_dict={x1: img_batch1,
                                               y1: label_batch1,
                                               x2: img_batch2,
                                               y2: label_batch2,
                                               keep_prob: dropout_rate})
                    writer.add_summary(s, epoch * train_batches_per_epoch1 + step)
                    


            print("{}: Start validation".format(datetime.now()))
            sess.run(testing_initalizer1)
            sess.run(testing_initalizer2)
            test_acc = 0
            test_count = 0
            test_acc1 = 0
            test_count1 = 0
            test_acc2 = 0
            test_count2 = 0            
            for _ in range(test_batches_per_epoch1):
                img_batch1, label_batch1 = sess.run(test_next_batch1)
                img_batch2, label_batch2 = sess.run(test_next_batch2)                
                f11,a11,correct_f11,acc1 = sess.run([f1,a1,correct_f1,accuracy1], feed_dict={x1: img_batch1,
                                                    y1: label_batch1,
                                                    keep_prob: dropout_rate})
                   
                print 'f11..................................'                                    
                print f11                                                                       
                print 'a11..................................'                                    
                print a11                                                                                        
                print 'correct_f11..................................'                                    
                print correct_f11 
                print 'acc1..................................'                                    
                print acc1                                    
                test_acc1 += acc1
                test_count1 += 1
                f12,a12,correct_f12,acc2= sess.run([f2,a2,correct_f2,accuracy2], feed_dict={x2: img_batch2,
                                                    y2: label_batch2,
                                                    keep_prob: dropout_rate})                                             
                print 'f12..................................'                                    
                print f12                                                                       
                print 'a12..................................'                                    
                print a12                                                                                        
                print 'correct_f12..................................'                                    
                print correct_f12 
                print 'acc2..................................'                                    
                print acc2                                    
                test_acc2 += acc2
                test_count2 += 1
                print test_count2
                correct_ff,acc= sess.run([correct_f,accuracy], feed_dict={ x1: img_batch1,
                                                    y1: label_batch1,
                                                    x2: img_batch2,
                                                    y2: label_batch2,
                                                    keep_prob: dropout_rate})  
                print 'correct_ff..................................'                                    
                print correct_ff 
                print 'acc..................'
                print acc
                test_acc += acc
                test_count += 1                               
            try:
                test_acc1 /= test_count1
                test_acc2 /= test_count2
                test_acc /= test_count                   
            except:
                print('ZeroDivisionError!')
            print("{}: Validation Accuracy1 = {:.4f}".format(datetime.now(), test_acc1))
            print("{}: Validation Accuracy2= {:.4f}".format(datetime.now(), test_acc2))
            print("{}: Validation Accuracy= {:.4f}".format(datetime.now(), test_acc))
    
            print("{}: Saving checkpoint of model...".format(datetime.now()))
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            # this epoch is over
            print("{}: Epoch number: {} end".format(datetime.now(), epoch + 1))


if __name__ == '__main__':
    main()
