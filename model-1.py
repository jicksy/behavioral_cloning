import csv
import cv2
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# 1. Load Data 
lines = []
def load_data(path):    
    with open(path+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile, )
        next(reader)
        for line in reader:
            lines.append(line)
    return lines




path = '/opt/carnd_p3/data/'

# loading the image paths from csv
lines = load_data(path)
print(len(lines))

images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[0]
        #print(source_path)
        filename = source_path.split('/')[-1]
        current_path = '/opt/carnd_p3/data/IMG/' + filename
        #print(current_path)
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
    
# Data augmentation
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    
    
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print(y_train[1])


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
# Normalize and mean center
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))


model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 2)

model.save('model.h5')
