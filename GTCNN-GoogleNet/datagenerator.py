# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.data import Dataset


VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):
    def __init__(self, images, labels, batch_size, num_classes, mode,output_buffer_size,image_format, shuffle=True,buffer_size=1000):
        self.img_paths = images
        self.labels = labels
        self.data_size = len(self.labels)        
        self.num_classes = num_classes
        self.image_format = image_format

        if shuffle:
            self._shuffle_lists()

        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
        if mode == 'training':
            data = data.map(self._parse_function_train)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)
        data = data.prefetch(output_buffer_size)    
        data = data.batch(batch_size)
        self.data = data
        
        


    def _shuffle_lists(self):
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])


    def _parse_function_train(self, filename, label):
        one_hot = tf.one_hot(label, self.num_classes)
        img_string = tf.read_file(filename)
        if self.image_format == "jpg":
            img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        elif self.image_format == "png":
            img_decoded = tf.image.decode_png(img_string, channels=3)
        else:
            print("Error! Can't confirm the format of images!")
        img_decode = tf.cast(img_decoded, tf.float32)    
        img_resized = tf.image.resize_images(img_decode, [224,224])
        img_centered = tf.subtract(img_resized, VGG_MEAN )
        img_bgr = img_centered[:, :, ::-1]
        return img_bgr, one_hot

    def _parse_function_inference(self, filename, label):
        one_hot = tf.one_hot(label, self.num_classes)
        img_string = tf.read_file(filename)
        if self.image_format == "jpg":
            img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        elif self.image_format == "png":
            img_decoded = tf.image.decode_png(img_string, channels=3)
        else:
            print("Error! Can't confirm the format of images!")
        img_decode = tf.cast(img_decoded, tf.float32)                
        img_resized = tf.image.resize_images(img_decode, [224,224])
        img_centered = tf.subtract(img_resized, VGG_MEAN )
        img_bgr = img_centered[:, :, ::-1]
        return img_bgr, one_hot