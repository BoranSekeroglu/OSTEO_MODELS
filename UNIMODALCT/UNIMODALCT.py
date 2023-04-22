import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

test_c = []
test_o = []

# read test image...
test_image_C = "26246471.jpg"
img = image.load_img(test_image_C,target_size=(80,60,3))
img = image.img_to_array(img)
norm = img/255
test_c.append(norm)
X = np.array(test_c)

test_image_O = "16204172.jpg"
img = image.load_img(test_image_O,target_size=(80,60,3))
img = image.img_to_array(img)
norm = img/255
test_o.append(norm)
X1 = np.array(test_o)
     
# load model and weights...
loaded_model = keras.models.load_model("UNIMODALCT.h5") 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
loaded_model.summary()

# perform predictions...
y_pred = loaded_model.predict(X)   

print(test_image_C, ' --> Osteoporosis:', y_pred) if np.round(y_pred) == 1 else print(test_image_C, ' --> Healthy:', y_pred)

y_pred = loaded_model.predict(X1)   

print(test_image_O, '--> Osteoporosis:', y_pred) if np.round(y_pred) == 1 else print(test_image_C, ' --> Healthy:', y_pred)
