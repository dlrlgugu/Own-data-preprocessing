import numpy as np
import os
import scipy
from PIL import Image
from scipy import ndimage

def own_data_preprocessing():
    path = 'C:\\gugu\\project\\Own data preprocessing\\dogs'
    resize_path = 'C:\\gugu\\project\\Own data preprocessing\\resize'

    img_list = os.listdir(path)
    row , col = 64 , 64

    number_of_samples = len(img_list)

    for file in img_list:
        if file.endswith('jpeg'):
            im = Image.open(path+'\\'+file)
            img = im.resize((row,col))
            img.save(resize_path+'\\'+file,"jpeg")

    resize_img_list = os.listdir(resize_path)
    img_matrix = np.array([np.array(Image.open(resize_path+'\\'+img)) for img in resize_img_list]
                          ,'f')

    label = np.ones((number_of_samples,),dtype=int)

    train_x = img_matrix[0:500]
    label[0:500]=1
    train_y = label[0:500]
    train_y = train_y.reshape((1,train_y.shape[0]))

    test_x = img_matrix[501:676]
    label[501:676] = 1
    test_y = label[501:676] # 1-d vector (175,)
    test_y = test_y.reshape((1,test_y.shape[0])) #(1,175)

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


    #img_matrix = np.array([np.array(Image.open(path+'\\'+img)) for img in img_list],'f')


own_data_preprocessing()
                          
