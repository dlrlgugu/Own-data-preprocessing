import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from PIL import Image
from scipy import ndimage
import random
from sklearn.utils import shuffle

def own_data_preprocessing():
    path = 'C:\\gugu\\project\\Own data preprocessing\\input_data'
    resize_path = 'C:\\gugu\\project\\Own data preprocessing\\resize'

    img_list = os.listdir(path)
    row , col = 64 , 64

    number_of_samples = len(img_list)

    for file in img_list:
        im = Image.open(path+'\\'+file)
        img = im.resize((row,col))
        img.save(resize_path+'\\'+file,"JPEG")#what's wrong with JPG??

    resize_img_list = os.listdir(resize_path)
    img_matrix = np.array([np.array(Image.open(resize_path+'\\'+img))
                           for img in resize_img_list],'f') # no flatten.

    label = np.ones((number_of_samples,),dtype=int)
    label[0:232]=0
    label[233:465]=1
    #assumed pictures are sorted. or else everything will mess up
    #Training set label

        
    data , Label = shuffle(img_matrix,label,random_state=2)
    #train_data = [data,Label]
    #train_data = train_data/255.

    train_x = data[0:400]
    train_y = Label[0:400]
    train_y = train_y.reshape((1,train_y.shape[0]))

    test_x = data[401:465]
    test_y = Label[401:465] # 1-d vector (175,)
    test_y = test_y.reshape((1,test_y.shape[0]))

    #image = train_data[0][25].reshape(64,64,3)
    #plt.imshow(train_data[0][1])
    #plt.show()
    #print(train_data[0][25])
    #print(train_data[1][25])
    

    """
    #without shuffling.
    
    train_x = img_matrix[0:200]+img_matrix[233:433]
    #it just add list not append..
    train_y = label[0:200]+label[233:433]
    train_y = train_y.reshape((1,train_y.shape[0]))
    
    test_x=img_matrix[201:232]+img_matrix[434:465]
    test_y=label[201:232]+label[434:465]
    test_y=test_y.reshape((1,test_y.shape[0]))

    index = random.randrange(1,400)
    test_img = train_x[index]/255.
    print(label[index])
    plt.imshow(test_img)
    plt.show()
    """
    
    """
    #work example
    index = random.randrange(1,465)
    test_img = img_matrix[index]/255.
    print(label[index])
    plt.imshow(test_img)
    plt.show()
    """

    """
    list sliding add example
    a = [i for i in range(1,10)]
    b = a[0:2] + a[5:8]    
    """
    
    """
    ex)data set setting
    
    train_x = img_matrix[0:232]
    label[0:232]=1
    train_y = label[0:500]
    train_y = train_y.reshape((1,train_y.shape[0]))

    test_x = img_matrix[501:676]
    label[501:676] = 1
    test_y = label[501:676] # 1-d vector (175,)
    test_y = test_y.reshape((1,test_y.shape[0])) #(1,175)
    """

    """
    example)
    q=np.array([1,1,1])
    q.shape #(3,)
    q #[1,1,1]

    q=q.reshape((1,3))
    q.shape #(3,1)
    q #[[1,1,1]]

    """
   
    #print(train_x.shape) #(500, 64, 64, 3)
    #print(train_y.shape) #(1, 500)
    

    return train_x,train_y,test_x,test_y



own_data_preprocessing()
                          
