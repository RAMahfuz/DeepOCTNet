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
from keras.applications import vgg19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint

conv_base = vgg19.VGG19(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3))

import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
base_dir ='D:/Datasets/OCT/OCT2017_NOR_3_class'
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
test_datagen = ImageDataGenerator(rescale=1./255)

model.load_weights("vgg19_weights_amd_cnv_dme.best.hdf5")

from keras import optimizers
model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])



test_generator = test_datagen.flow_from_directory(test_dir,target_size=(224, 224),batch_size=10,class_mode=None, shuffle = False)
probabilities = model.predict_generator(test_generator, 726)
prob=probabilities[0:726,:]


#Creating true label
import pandas as pd
from sklearn.preprocessing import label_binarize
labels = np.array([0] * 242 + [1] * 242 + [2] * 242)
y_test = label_binarize(labels, classes=[0, 1, 2])



#Calculating predicted label
y_cal = np.argmax(prob, axis = 1)
y_score = label_binarize(y_cal, classes=[0, 1, 2])


#Confusion Matrix and test accuracy
from sklearn import metrics
from sklearn.metrics import accuracy_score

print(metrics.accuracy_score(labels,y_cal))
confusion = metrics.confusion_matrix(labels,y_cal)
print(confusion)

from pycm import ConfusionMatrix
cm = ConfusionMatrix(actual_vector=labels, predict_vector=y_cal)
print(cm)

#https://hackernoon.com/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier-2ecc6c73115a
#import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 2
n_classes = 3

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average(area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average(area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)


plt.plot(fpr[0], tpr[0], color='blue', lw=lw, label='AMD (area = {:.2f})'.format(roc_auc[0]))
plt.plot(fpr[1], tpr[1], color='green', lw=lw, label='CNV (area = {:.2f})'.format(roc_auc[1]))
plt.plot(fpr[2], tpr[2], color='red', lw=lw, label='DME (area = {:.2f})'.format(roc_auc[2]))

   
    
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curves')
plt.legend(loc="lower right")
plt.show()




# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average(area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average(area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

plt.plot(fpr[0], tpr[0], color='blue', lw=lw, label='AMD (area = {:.2f})'.format(roc_auc[0]))
plt.plot(fpr[1], tpr[1], color='green', lw=lw, label='CNV (area = {:.2f})'.format(roc_auc[1]))
plt.plot(fpr[2], tpr[2], color='red', lw=lw, label='DME (area = {:.2f})'.format(roc_auc[2]))


plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()





print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(labels,y_cal)
np.set_printoptions(precision=2)
class_names=['CNV', 'DME', 'DRUSEN', 'NORMAL']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()