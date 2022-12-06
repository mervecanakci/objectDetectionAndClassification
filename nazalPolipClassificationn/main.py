import matplotlib.pyplot as plt
import seaborn as sns
import self as self

from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam_v2

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

import tensorflow as tf

import cv2
import os

import numpy as np

labels = ['withNP', 'withoutNP']
test_frac = 0.2
data_path = 'dataset/data'

d_len = len(os.listdir(os.path.join(data_path, labels[0])))

img_size = 224
def get_data(data_dir, start, finish):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        if finish == 0:
            ims = os.listdir(path)[start:]
        else:
            ims = os.listdir(path)[:finish]
        for img in ims:
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # resmin boyutunu değiştirme
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

train = get_data('dataset/data/', 0, -int(d_len * 0.2))
val = get_data('dataset/data/', -int(d_len * 0.2), 0)

l = []
for i in train: # egitim baslaması
    if(i[1] == 0):# eğitimin içinde gezsin i
        l.append("withNp")# sonuç 0 ise polip var
    else:
        l.append("withoutNP")
sns.set_style('darkgrid')
sns.countplot(l)

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train: # eğitim için özellik ve label çıkarma
  x_train.append(feature)
  y_train.append(label)

for feature, label in val: # validation için özellik ve label çıkarma
  x_val.append(feature)
  y_val.append(label)


# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
#resmin verilerini çıkardık

datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

opt = adam_v2.Adam(lr=0.000001)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

epoch_len =100

history = model.fit(x_train,y_train,epochs = epoch_len , validation_data = (x_val, y_val))
"""
y_predict=np.asarray(model.predict(x_val))

y_val = np.argmax(y_val, axis=0)
y_predict = np.argmax(y_predict, axis=1)
self._data=[]
self._data.append({
            'val_recall': recall_score(y_val, y_predict),
            'val_precision': precision_score(y_val, y_predict),
        })
print(self)
"""
acc = history.history['accuracy'] #tp
val_acc = history.history['val_accuracy']#tn
loss = history.history['loss']#fp
val_loss = history.history['val_loss']#fn

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
y_pred1 = model.predict(x_train)
y_pred_a = np.argmax(y_pred1, axis=1)

# Print f1, precision, and recall scores
print("precision score: ", precision_score(y_train, y_pred_a , average="macro"))
print("recall score: ", recall_score(y_train, y_pred_a , average="macro"))
# print(" f1 score: ", f1_score(y_train, y_pred , average="macro"))


epochs_range = range(epoch_len)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
"""
plt.subplot(2, 2, 2)S
plt.plot(epochs_range, acc/(acc+val_loss), label='Training Loss')
plt.plot(epochs_range,acc/(acc+val_acc), label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

"""
predictions = model.predict(x_val)
predictions = np.argmax(predictions,axis=1)
print(len(x_val))
print(len(predictions))
print(classification_report(y_val, predictions, target_names = ['withNP (Class 0)','withoutNP (Class 1)']))