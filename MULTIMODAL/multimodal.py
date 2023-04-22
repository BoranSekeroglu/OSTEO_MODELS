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
test_image_MRI = "mri0.jpg"
img = image.load_img(test_image_MRI,target_size=(80,60,3))
img = image.img_to_array(img)
norm = img/255
test_c.append(norm)
X = np.array(test_c)

test_image_CT = "ct1.jpg"
img = image.load_img(test_image_CT,target_size=(80,60,3))
img = image.img_to_array(img)
norm = img/255
test_o.append(norm)
X1 = np.array(test_o)
     
# load model and weights...
loaded_model = keras.models.load_model("multimodal.h5") 
# evaluate loaded model on test data
loaded_model.compile(optimizer='adam', metrics=['accuracy'])
loaded_model.summary()

y_pred = loaded_model.predict([X, X1])

print(test_image_MRI, ' --> Osteoporosis:', y_pred[0]) if np.round(y_pred[0]) == 1 else print(test_image_MRI, ' --> Healthy:', y_pred[0])
print(test_image_CT, ' --> Osteoporosis:', y_pred[1]) if np.round(y_pred[1]) == 1 else print(test_image_CT, ' --> Healthy:', y_pred[1])

