print("Loading .............")

from tensorflow import keras
import cv2
import numpy as np
import tensorflow_hub as hub

from tkinter import Tk
from tkinter.filedialog import askopenfilename

import os
import time



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
custom_objects = {'KerasLayer': hub.KerasLayer}

absolute_path = os.path.dirname(__file__)
model_path= os.path.join(absolute_path,"MobileModel.h5")

classes={'Transistor':0,'Resistor':1,'Capacitor':2,'LED':3,'Chip':4,'Misc':5}

model=keras.models.load_model(model_path,custom_objects=custom_objects)

root = Tk()
root.withdraw() 


while True:
 
 cont=input("Press Any key to Continue || Press 'q' to quit : ")
 if cont=='q':
   break
   print("Closing in 5 seconds!")
   time.sleep(5)

 file_path = askopenfilename()

 image = cv2.imread(str(file_path))
 resized_image = cv2.resize(image, (128,128))
 resized_image=resized_image.reshape(1,128,128,3)
 resized_image=np.array(resized_image)
 resized_image=resized_image/255

 output=model.predict(resized_image)
 


 class_index=np.argmax(output)
 for i in classes:
  if classes[i]==class_index:
       print('-----------------------------------------------------------------------------------------')
       print("The result is ",i)
       print('-----------------------------------------------------------------------------------------')


 


