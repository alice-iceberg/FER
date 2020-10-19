from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGE_SIZE = [224, 224]
EPOCHS_NUM = 8
MODEL_NAME = "mixed_exp2_model.h5"

train_path = 'train_rgb_224_CO'
valid_path = 'test_rgb_224_CO'

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet',
            include_top=False)  # [3] for RGB channel, [1] for gray images

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

# check how many labels we have
folders = glob('train_rgb_224_CO/*')

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

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

test_set = test_datagen.flow_from_directory(valid_path,
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

model.save(MODEL_NAME)

imgs, labels = next(training_set)

# Images Classes with index
print(training_set.class_indices)

# Print the Target names
target_names = []
for key in training_set.class_indices:
    target_names.append(key)

print('Target names: ')
print(target_names)
Y_pred = model.predict(test_set)
y_pred = np.argmax(Y_pred, axis=1)


def plot_confusion_matrix(cm_inside,
                          target_names_inside,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm_inside:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names_inside: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    """
    import itertools

    accuracy = np.trace(cm_inside) / np.sum(cm_inside).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_inside, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names_inside is not None:
        tick_marks = np.arange(len(target_names_inside))
        plt.xticks(tick_marks, target_names_inside, rotation=45)
        plt.yticks(tick_marks, target_names_inside)

    if normalize:
        cm_inside = cm_inside.astype('float') / cm_inside.sum(axis=1)[:, np.newaxis]

    thresh = cm_inside.max() / 1.5 if normalize else cm_inside.max() / 2
    for i, j in itertools.product(range(cm_inside.shape[0]), range(cm_inside.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm_inside[i, j]),
                     horizontalalignment="center",
                     color="white" if cm_inside[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm_inside[i, j]),
                     horizontalalignment="center",
                     color="white" if cm_inside[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# Confusion Matrix

print('Confusion Matrix')
cm = confusion_matrix(test_set.classes, y_pred)
plot_confusion_matrix(cm, target_names)
# Print Classification Report
print('Classification Report')
print(classification_report(test_set.classes, y_pred, target_names=target_names))

# #################################################TEST a single image
# print('Testing')
# model = load_model('facefeatures_third_exp_model.h5')
# model.summary()
#
# file = 'HA3.jpg'
#
# img = cv.imread(file)
#
# test_image = image.img_to_array(img)
# test_image = np.expand_dims(test_image, axis=0)
# pred = model.predict(test_image)
# print(pred, labels[np.argmax(pred)])
