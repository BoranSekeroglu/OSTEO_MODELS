import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

train_image = []
train = []

osteo = '035516(119).jpg'
control = '15812904_t1_tse_sag_7.jpg'

img = image.load_img(osteo,target_size=(80,60,3))
img = image.img_to_array(img)
resized = img/255
train_image.append(resized)
train.append(1)
img = image.load_img(control,target_size=(80,60,3))
img = image.img_to_array(img)
resized = img/255
train_image.append(resized)
train.append(0)
    

y = np.array(train)
X = np.array(train_image)



inputs = keras.Input(shape=(80,60, 3), name="img")
block1A = layers.Conv2D(32, 3, padding='valid', activation="relu",name = "a1")(inputs)
block1Ar = layers.BatchNormalization()(block1A)
block1Ao = layers.MaxPooling2D(2)(block1Ar)
block1B = layers.Conv2D(32, 3, padding='valid', activation="relu",name = "a2")(block1Ao)
block1Br = layers.BatchNormalization()(block1B)
block1Bo = layers.MaxPooling2D(2)(block1Br)
   
block2A = layers.Conv2D(32, 5, strides = (2,2), padding='same', name = "b1", activation="relu")(inputs)
block2Ar = layers.BatchNormalization()(block2A)
block2B = layers.Conv2D(32, 5, strides = (2,2), padding='valid', name = "b2", activation="relu")(block2Ar)
block2Br = layers.BatchNormalization()(block2B)


add1 = layers.add([block1Bo, block2Br],name = "add_layer")
add1c = layers.Conv2D(64, 5, activation="relu", strides = (1,1), padding='same', name = "add_out")(add1)

fl = layers.Flatten()(add1c)

d1 = layers.Dense(32, activation="relu")(fl)

d2 = layers.Dense(16, activation="relu")(d1)

outputs = layers.Dense(1, activation="sigmoid")(d2)

model = keras.Model(inputs, outputs, name="uni")

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, y, epochs= 5, batch_size=16)  #validation_data = (X_val, y_val),
LO, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

model.save("UNIMODALCT.h5")
print("Model Saved...")

