 # -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:31:18 2019

@author: ramah
"""

# -*- coding: utf-8 -*-
"""
Validation accuracy:   64 %
Created on Wed Jan 23 01:56:10 2019

@author: ramah
"""

from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint

conv_base = vgg16.VGG16(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3))


import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
base_dir ='..../OCT2017_NOR_3_class'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir, 'test')

from keras import models
from keras import layers


x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(3, activation='softmax')(x)
model = models.Model(inputs=conv_base.input, outputs=predictions)





from keras import optimizers
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_dir,target_size=(224, 224),batch_size=10,class_mode='categorical')
label_map=(train_generator.class_indices)
print(label_map)
validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(224, 224),batch_size=10,class_mode='categorical')

model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])

filepath = "vgg16_weights_amd_cnv_dme.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose =1, save_best_only=True, mode = 'max')
callbacks_list = [checkpoint]

history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=100, callbacks=callbacks_list, verbose =1, validation_data=validation_generator,validation_steps=100)

print("Saved model to disk")




import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc =history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']



epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label= 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


np.savetxt('Accuracy.dat',acc)
np.savetxt('loss.dat',loss)
np.savetxt('Vvalidation_Accuracy.dat',val_acc)
np.savetxt('Validation_loss.dat',val_loss)

