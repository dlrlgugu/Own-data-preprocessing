import numpy as np
import os
import scipy
from PIL import Image
from scipy import ndimage

def rename_images():
    i=0
    for file in os.listdir():
        
        filename = os.fsdecode(file)
        if file.endswith('jpeg'):
            os.rename(filename,"dog_"+str(i)+".jpg")
            
        i = i+1
        
rename_images()
            
