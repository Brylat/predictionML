from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import pickle
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

with open("./first_model.mdm", "rb") as f:
    classifier = pickle.load(f)

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
result2 = classifier.predict_proba(test_image/255);
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(result)    
print(prediction)
print(result2)