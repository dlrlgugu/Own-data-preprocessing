import tensorflow as tf
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *
from own_data_preprocessing import own_data_preprocessing

train_x_orig, train_y, test_x_orig, test_y = own_data_preprocessing()

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
# The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

layers_dims=[12288,20,10,15,5,3,1] # 4 layer.

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 2000, print_cost=False):
    np.random.seed(1)

    costs=[]

    parameters=initialize_parameters_deep(layers_dims)

    for i in range(0,num_iterations):
        AL,caches = L_model_forward(X,parameters)
        cost = compute_cost(AL,Y)
        grads=L_model_backward(AL,Y,caches)
        parameters=update_parameters(parameters,grads,learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

import time
start=time.time()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
with tf.device('/gpu:0'):   
    parameters = L_layer_model(train_x, train_y, layers_dims,
                               num_iterations = 500, print_cost = True)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)
#print_mislabeled_images(classes, test_x, test_y, pred_test)

done = time.time()
print("it took a "+str(done-start)+" seconds")

test_image = ["cat.jpg","cat1.jpg","cat2.jpg","cat3.jpg","cat4.jpg",
              "dog1.jpg","dog2.jpg","dog3.jpg"]
my_label_y = [0,0,0,0,0,1,1,1]
for i in test_image:
    image = np.array(ndimage.imread(i, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
    my_image = my_image/255.
    my_predicted_image = predict(my_image, my_label_y, parameters)

    plt.imshow(image)
    print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"")
    print (my_predicted_image)

#iteration 3000 results = 0,0,0,1,1,0,1,0(5 out of 8)
#Accuracy: 1.0 Accuracy: 0.625
    
#iteration 1700 results = 0,0,0,1,1,0,1,0(5 out of 8)
#Accuracy: 0.965 Accuracy: 0.5625

#iteration 500  results = 0,0,0,0,0,0,1,0(6 out of 8)
#Accuracy: 0.7775 Accuracy: 0.640625
"""
my_image = "my_image.jpg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_image = my_image/255.
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
"""
