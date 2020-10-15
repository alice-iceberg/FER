from keras.models import load_model
import cv2 as cv
from keras.preprocessing import image
import numpy as np
from main import labels


print('Testing')
model = load_model('facefeatures_first_exp_model.h5')
model.summary()

file = 'SA2.jpg'

img = cv.imread(file)

test_image = image.img_to_array(img)
test_image = np.expand_dims(test_image, axis=0)
pred = model.predict(test_image)
print(pred, labels[np.argmax(pred)])

