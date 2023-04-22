import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

def preprocess(path):
    img = image.load_img(path,target_size=(80,60,3))
    img = image.img_to_array(img)
    resized = img/255
    return(resized)    


trainMR = []
targetMR = []
trainCT = []
targetCT = []
MR0 = 'mri0.jpg'
MR1 = 'mri1.jpg'
CT0 = 'ct0.jpg'
CT1 = 'ct1.jpg'


trainMR.append(preprocess(MR0))
targetMR.append(0)
trainMR.append(preprocess(MR1))
targetMR.append(1)
trainCT.append(preprocess(CT0))
targetCT.append(0)
trainCT.append(preprocess(CT1))
targetCT.append(1)

XT = np.array(trainMR)
X2T = np.array(trainCT)
yT = np.array(targetMR)
y2T = np.array(targetCT)

   #mri block
inputs1 = keras.Input(shape=(80, 60, 3), name="img")   
block1A = layers.Conv2D(32, 3, padding='same', activation="relu",name = "a1")(inputs1)
block1Ar = layers.BatchNormalization()(block1A)
block1Ao = layers.MaxPooling2D(2)(block1Ar)
block1B = layers.Conv2D(32, 3, padding='same', activation="relu",name = "a2")(block1Ao)
block1Br = layers.BatchNormalization()(block1B)
block1Bo = layers.MaxPooling2D(2)(block1Br)
   
block2A = layers.Conv2D(32, 5, strides = (2,2), padding='same', name = "b1", activation="relu")(inputs1)
block2Ar = layers.BatchNormalization()(block2A)
block2B = layers.Conv2D(32, 5, strides = (2,2), padding='same', name = "b2", activation="relu")(block2Ar)
block2Br = layers.BatchNormalization()(block2B)
   
add1 = layers.add([block1Bo, block2Br],name = "add_layer")
add1c = layers.Conv2D(64, 5, activation="relu", strides = (1,1), padding='same', name = "add_out")(add1)
   
fl = layers.Flatten()(add1c)

d1 = layers.Dense(32, activation="relu")(fl)
   
d2 = layers.Dense(16, activation="relu")(d1)
   
outputs = layers.Dense(1, activation="sigmoid", name = "o1")(d2)
   
   
   
 #ct block
inputs2 = keras.Input(shape=(80, 60, 3))   
block1Au = layers.Conv2D(32, 3, padding='same', activation="relu",name = "a1c")(inputs2)
block1Aru = layers.BatchNormalization()(block1Au)
block1Aou = layers.MaxPooling2D(2)(block1Aru)
block1Bu = layers.Conv2D(32, 3, padding='same', activation="relu",name = "a2c")(block1Aou)
block1Bru = layers.BatchNormalization()(block1Bu)
block1Bou = layers.MaxPooling2D(2)(block1Bru)
   
  
block2Au = layers.Conv2D(32, 5, strides = (2,2), padding='same', name = "b1c", activation="relu")(inputs2)
block2Aru = layers.BatchNormalization()(block2Au)
block2Bu = layers.Conv2D(32, 5, strides = (2,2), padding='same', name = "b2c", activation="relu")(block2Aru)
block2Bru  = layers.BatchNormalization()(block2Bu)
   
add1u = layers.add([block1Bou, block2Bru],name = "add_layerc")
add1cu = layers.Conv2D(64, 5, activation="relu", strides = (1,1), padding='same', name = "add_ouct")(add1u)
   
flu = layers.Flatten()(add1cu)

d1u = layers.Dense(32, activation="relu")(flu)
   
d2u = layers.Dense(16, activation="relu")(d1u)
   
outputsu = layers.Dense(1, activation="sigmoid", name = "o2")(d2u)

 
model = keras.Model(inputs=[inputs1, inputs2], outputs=[outputs, outputsu])

model.summary()

losses = {
	"o1": "binary_crossentropy",
	"o2": "binary_crossentropy",
    }
lossWeights = {"o1": 1.0, "o2": 1.0}
 
model.compile(loss=losses, loss_weights = lossWeights, optimizer='adam', metrics=['accuracy'])

history = model.fit([XT,X2T], [yT,y2T], epochs=5, batch_size=8) 
y_pred = model.predict([XT, X2T])
print(y_pred)
y_pred = np.array(y_pred)

print(MR0, ' --> Osteoporosis:', y_pred[0,0]) if np.round(y_pred[0,0]) == 1 else print(MR0, ' --> Healthy:', y_pred[0,0])
print(MR1, ' --> Osteoporosis:', y_pred[0,1]) if np.round(y_pred[0,1]) == 1 else print(MR1, ' --> Healthy:', y_pred[0,1])
print(CT0, ' --> Osteoporosis:', y_pred[1,0]) if np.round(y_pred[1,0]) == 1 else print(CT0, ' --> Healthy:', y_pred[1,0])
print(CT1, ' --> Osteoporosis:', y_pred[1,1]) if np.round(y_pred[1,1]) == 1 else print(CT1, ' --> Healthy:', y_pred[1,1])


