from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import classification_report, confusion_matrix
import os
from keras.models import load_model
import cv2 as cv
from keras.preprocessing import image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGE_SIZE = [224, 224]
EPOCHS_NUM = 10

train_path = 'train_rgb_224'
valid_path = 'test_rgb_224'

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet',
            include_top=False)  # [3] for RGB channel, [1] for gray images

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

# check how many labels we have
folders = glob('train_rgb_224/*')

# our layers, we can add more
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# data augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('train_rgb_224',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

test_set = test_datagen.flow_from_directory('test_rgb_224',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical',
                                            shuffle=False)

# fit the model
r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=EPOCHS_NUM,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

model.save('facefeatures_first_exp_model.h5')

imgs, labels = next(training_set)

# Images Classes with index
print(training_set.class_indices)


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Print the Target names

target_names = []
for key in training_set.class_indices:
    target_names.append(key)

print('Target names: ')
print(target_names)

# Confusion Matrix

Y_pred = model.predict(test_set)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(test_set.classes, y_pred)
plot_confusion_matrix(cm, target_names, title='Confusion Matrix')

# Print Classification Report
print('Classification Report')
print(classification_report(test_set.classes, y_pred, target_names=target_names))


# #################################################TEST
print('Testing')
model = load_model('facefeatures_first_exp_model.h5')
model.summary()

file = 'SA2.jpg'

img = cv.imread(file)

test_image = image.img_to_array(img)
test_image = np.expand_dims(test_image, axis=0)
pred = model.predict(test_image)
print(pred, labels[np.argmax(pred)])

