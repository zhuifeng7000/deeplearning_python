# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:29:21 2018

@author: Administrator
"""

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))