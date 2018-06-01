import numpy as np
import os
import scipy
from PIL import Image
from scipy import ndimage

def rename_images():
    i=0
    for file in os.listdir():
        
        filename = os.fsdecode(file)
        if file.endswith('jpg'):
            os.rename(filename,"cat_"+str(i)+".jpg")
            """
            im = Image.open(cat_path+'\\'+file)
            img = im.resize((row,col))
            os.rename(file,file.replace(file,"cat_"+str(iter)))
            img.save(cat_resize_path+'\\'+file,"jpeg")
            #+"cat_"+str(iter)
        print(iter)
            """
        i = i+1
        
rename_images()
            
