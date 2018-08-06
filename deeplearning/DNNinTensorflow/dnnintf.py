# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 16:06:45 2018

@author: 赵智广

dnn in tensorflow
增加了regulartion，防止overfit
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

##简单程序
#y_hat = tf.constant(36,name='y_hat')
#y=tf.constant(39,name='y')
#
#loss = tf.Variable((y-y_hat)**2,name='loss')
#
#init = tf.global_variables_initializer()
#
#with tf.Session() as session:
#    session.run(init)
#    print(session.run(loss))
   
##placeholder 
#x = tf.placeholder(tf.int64,name='x')
##with tf.Session() as session:
##    print(session.run(2*x,feed_dict={x:3}))
#sess = tf.Session()
#print(sess.run(2*x,feed_dict={x:3}))
#sess.close()


# GRADED FUNCTION: linear_function
def linear_function():
    np.random.seed(1)
    #define tensor
    X = tf.constant(np.random.randn(3,1),name="X")
    W = tf.constant(np.random.randn(4,3),name="W")
    b = tf.constant(np.random.randn(4,1),name="b")
    
    #define operations
    Y = tf.add(tf.matmul(W,X),b)

    #run
    sess = tf.Session()
    result = sess.run(Y)
    
    #close session
    sess.close()
    
    return result

#print("result=",linear_function())

# GRADED FUNCTION: sigmoid
def sigmoid(z):
    #define tensor
    x = tf.placeholder(tf.float32,name="x")
    
    with tf.Session() as sess:
        result=sess.run(tf.sigmoid(x),feed_dict={x:z})
        
    return result


#print ("sigmoid(0) = " + str(sigmoid(0)))
#print ("sigmoid(12) = " + str(sigmoid(12)))

# GRADED FUNCTION: cost
def cost(logits,labels):
    Z = tf.placeholder(tf.float32,name="Z")
    Y = tf.placeholder(tf.float32,name="Y")
    
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z,labels=Y)
    
    with tf.Session() as sess:
        cost=sess.run(loss,feed_dict={Z:logits,Y:labels})
        
    return cost


#logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
#cost = cost(logits, np.array([0,0,1,1]))
#print(logits)

#----------------
#indices = [0, 1, 1]
#depth = 3
#
#
#with tf.Session() as session:
#    print(session.run(tf.one_hot(indices, depth,axis=0)))

#-------------------
    
# GRADED FUNCTION: one_hot_matrix
def one_hot_matrix(labels,C):
    C_tf = tf.constant(C)
    matrix = tf.one_hot(labels,C_tf,axis=0)
    with tf.Session() as sess:
        mat = sess.run(matrix)
        
    return mat

#labels = np.array([1,2,3,0,2,1])
#one_hot = one_hot_matrix(labels, C = 4)
#print ("one_hot = " + str(one_hot))


def ones(shape):
    tfOnes = tf.ones(shape)
    with tf.Session() as sess:
        resultM = sess.run(tfOnes)
        
    return resultM

#print(ones((3,4)))

def load():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # Normalize image vectors
    X_train = X_train_flatten/255.
    X_test = X_test_flatten/255.
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)
    
    print ("number of training examples = " + str(X_train.shape[1]))
    print ("number of test examples = " + str(X_test.shape[1]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    return X_train, Y_train, X_test, Y_test







#X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
#print(X_train_orig.shape)
#print(Y_train_orig.shape)
#print(X_test_orig.shape)
#print(Y_test_orig.shape)
#
##index=0
##plt.imshow(X_train_orig[index])
##print("y = "+str(np.squeeze(Y_train_orig[:,index])))
#
## Flatten the training and test images
#X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
#X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
## Normalize image vectors
#X_train = X_train_flatten/255.
#X_test = X_test_flatten/255.
## Convert training and test labels to one hot matrices
#Y_train = convert_to_one_hot(Y_train_orig, 6)
#Y_test = convert_to_one_hot(Y_test_orig, 6)
#
#print ("number of training examples = " + str(X_train.shape[1]))
#print ("number of test examples = " + str(X_test.shape[1]))
#print ("X_train shape: " + str(X_train.shape))
#print ("Y_train shape: " + str(Y_train.shape))
#print ("X_test shape: " + str(X_test.shape))
#print ("Y_test shape: " + str(Y_test.shape))


#linear-relu-linear-relu-linear-softmax
def model(X_train,Y_train,X_test,Y_test,learning_rate=0.0001,num_epochs=1500,
          minibatch_sizes=32,print_cost=True):
    
    ops.reset_default_graph() 
    #some values
    (n_x,m) = X_train.shape #64*64*3
    n_y = Y_train.shape[0]
    n_h1 = 25
    n_h2 = 12
    costs=[]
    tf.set_random_seed(1)
    seed=3
    
    #input placeholser
    X = tf.placeholder(tf.float32,shape=(n_x,None),name="X")
    Y = tf.placeholder(tf.float32,shape=(n_y,None),name="Y")
    
    #regularizer
    regularizer = tf.contrib.layers.l2_regularizer(0.1)
    #initialize parameters
    W1 = tf.get_variable("W1",[n_h1,n_x],initializer=tf.contrib.layers.xavier_initializer(seed=1),regularizer=regularizer)
    b1 = tf.get_variable("b1",[n_h1,1],initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2",[n_h2,n_h1],initializer=tf.contrib.layers.xavier_initializer(seed=1),regularizer=regularizer)
    b2 = tf.get_variable("b2",[n_h2,1],initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3",[n_y,n_h2],initializer=tf.contrib.layers.xavier_initializer(seed=1),regularizer=regularizer)
    b3 = tf.get_variable("b3",[n_y,1],initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    #forward propagation
    Z1 = tf.matmul(W1,X)+b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2,A1)+b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3,A2)+b3
    

    
    #cost
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    cost += reg_term
    
    #backforward prop adn upadte
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    #initialize all the variables
    init = tf.global_variables_initializer()
    
    #start session
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost=0
            num_minibatches = int(m/minibatch_sizes)
            seed = seed+1
            
            minibatches = random_mini_batches(X_train,Y_train,minibatch_sizes,seed)
            
            for minibatch in minibatches:
                (minibatch_X,minibatch_Y) = minibatch
                
                #execute the optimizer
                _,minibatch_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                
                epoch_cost += minibatch_cost/num_minibatches
                
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                
                #每100次计算一下正确率
                #caculate the correct predictions
                correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Y))
                #caculate accuracy on the set
                accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
        
                print("Train Accuracy:",accuracy.eval({X:X_train,Y:Y_train}))
                print("Test Accuracy:",accuracy.eval({X:X_test,Y:Y_test}))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                    
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()    
              
        #save the parameters
#        parameters_tf = {"W1": W1,
#                  "b1": b1,
#                  "W2": W2,
#                  "b2": b2,
#                  "W3": W3,
#                  "b3": b3}
        parameters=sess.run(parameters)
        
        #caculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Y))
        #caculate accuracy on the set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
        
        print("Train Accuracy:",accuracy.eval({X:X_train,Y:Y_train}))
        print("Test Accuracy:",accuracy.eval({X:X_test,Y:Y_test}))
        
    
    return parameters
        

def test(X_train,Y_train,X_test,Y_test,learning_rate=0.0001,num_epochs=1500,
          minibatch_sizes=32,print_cost=True):
    ops.reset_default_graph() 
    #some values
    (n_x,m) = X_train.shape #64*64*3
    n_y = Y_train.shape[0]
    n_h1 = 25
    n_h2 = 12
    costs=[]
    tf.set_random_seed(1)
    seed=3
    
    #input placeholser
    X = tf.placeholder(tf.float32,shape=(n_x,None),name="X")
    Y = tf.placeholder(tf.float32,shape=(n_y,None),name="Y")
    
    #regularizer
    regularizer = tf.contrib.layer.l2_regularizer(0.1)
    #initialize parameters
    W1 = tf.get_variable("W1",[n_h1,n_x],initializer=tf.contrib.layers.xavier_initializer(seed=1),regularizer=regularizer)
    b1 = tf.get_variable("b1",[n_h1,1],initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2",[n_h2,n_h1],initializer=tf.contrib.layers.xavier_initializer(seed=1),regularizer=regularizer)
    b2 = tf.get_variable("b2",[n_h2,1],initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3",[n_y,n_h2],initializer=tf.contrib.layers.xavier_initializer(seed=1),regularizer=regularizer)
    b3 = tf.get_variable("b3",[n_y,1],initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        param = sess.run(parameters)
        
        
    return param


def test_with_your_image(parameters):
    import scipy
    from PIL import Image
    from scipy import ndimage
    
    ## START CODE HERE ## (PUT YOUR IMAGE NAME) 
    my_image = "thumbs_up.jpg"
    ## END CODE HERE ##
    
    # We preprocess your image to fit your algorithm.
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
    my_image_prediction = predict(my_image, parameters)
    
    plt.imshow(image)
    print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
   





#################################     
if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load()
    parameters = model(X_train, Y_train, X_test, Y_test)
    
    #test_with_your_image(parameters)
    

