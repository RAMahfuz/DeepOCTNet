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
base_dir ='C:/Users/ramah/Documents/Research/Current_Projects/OCT/Datasets/OCT2017_NOR'
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
predictions = layers.Dense(4, activation='softmax')(x)
model = models.Model(inputs=conv_base.input, outputs=predictions)





from keras import optimizers

test_datagen = ImageDataGenerator(rescale=1./255)



model.load_weights("vgg16_weights_all.best.hdf5")

from keras import optimizers
model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])


test_generator = test_datagen.flow_from_directory(test_dir,target_size=(224, 224),batch_size=10,class_mode=None, shuffle = False)
probabilities = model.predict_generator(test_generator, 968)
prob=probabilities[0:968,:]


#Creating true label
import pandas as pd
from sklearn.preprocessing import label_binarize
labels = np.array([0] * 242 + [1] * 242 + [2] * 242 + [3] * 242)
y_test = label_binarize(labels, classes=[0, 1, 2, 3])



#Calculating predicted label
y_cal = np.argmax(prob, axis = 1)
y_score = label_binarize(y_cal, classes=[0, 1, 2, 3])


#Confusion Matrix and test accuracy
from sklearn import metrics
from sklearn.metrics import accuracy_score

print(metrics.accuracy_score(labels,y_cal))
confusion = metrics.confusion_matrix(labels,y_cal)
print(confusion)

from pycm import ConfusionMatrix
cm = ConfusionMatrix(actual_vector=labels, predict_vector=y_cal)
print(cm)



#Precision recall curve . The code is adapted from the following link
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


n_classes=4
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))



lw = 2

from itertools import cycle
import matplotlib.pyplot as plt
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
leg_labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
leg_labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
leg_labels.append('micro-average (area = {0:0.2f})'
              ''.format(average_precision["micro"]))


l, = plt.plot(recall[0], precision[0], color='turquoise', lw=2)
lines.append(l)
leg_labels.append('AMD (area = {1:0.2f})'
              ''.format(0, average_precision[0]))

l, = plt.plot(recall[1], precision[1], color='teal', lw=2)
lines.append(l)
leg_labels.append('CNV (area = {1:0.2f})'
              ''.format(1, average_precision[1]))

l, = plt.plot(recall[2], precision[2], color='cornflowerblue', lw=2)
lines.append(l)
leg_labels.append('DME (area = {1:0.2f})'
              ''.format(2, average_precision[2]))

l, = plt.plot(recall[3], precision[3], color='darkorange', lw=2)
lines.append(l)
leg_labels.append('NORMAL (area = {1:0.2f})'
              ''.format(3, average_precision[3]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Inception v3')
plt.legend(lines, leg_labels, loc=(0.2, 0.6), prop=dict(size=14))


plt.show()


# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0.8, 1)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(recall["micro"], precision["micro"],
         label='micro-average(area = {0:0.2f})'
               ''.format(average_precision["micro"]),
         color='deeppink', linestyle=':', linewidth=4)


plt.plot(recall[0], precision[0], color='blue', lw=lw, label='AMD (area = {:.2f})'.format(average_precision[0]))
plt.plot(recall[1], precision[1], color='green', lw=lw, label='CNV (area = {:.2f})'.format(average_precision[1]))
plt.plot(recall[2], precision[2], color='red', lw=lw, label='DME (area = {:.2f})'.format(average_precision[2]))
plt.plot(recall[3], precision[3], color='cyan', lw=lw, label='NORMAL (area = {:.2f})'.format(average_precision[3]))

plt.xlabel('Recall', fontsize =14, fontweight ='bold')
plt.ylabel('Precision', fontsize =14, fontweight ='bold')
plt.title('VGG16', fontsize =15, fontweight ='bold')
plt.legend(loc=(0.4, 0.05),)
plt.show()






#Plotting ROC curve. The code is mainly adapted from the following link
#https://hackernoon.com/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier-2ecc6c73115a
#import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 2
n_classes = 4

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
plt.figure(3)
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
plt.plot(fpr[3], tpr[3], color='cyan', lw=lw, label='NORMAL (area = {:.2f})'.format(roc_auc[3]))


    
    
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curves')
plt.legend(loc="lower right")
plt.show()




# Zoom in view of the upper left corner.
plt.figure(4)
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
plt.plot(fpr[3], tpr[3], color='cyan', lw=lw, label='NORMAL (area = {:.2f})'.format(roc_auc[3]))

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
class_names=['AMD', 'CNV', 'DME', 'NORMAL']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()