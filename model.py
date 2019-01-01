import os
import csv

# folder where data is contained
path = '/opt/carnd_p3/data/'

# read data and store it in lines
lines = []
with open(path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # skip first row
    next(reader)
    for line in reader:
        lines.append(line)


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Split data into tran and validation samples, 20% of data is used for validation samples
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import cv2
import numpy as np
import sklearn

# Generator function
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # center image
                name = '/opt/carnd_p3/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                # convert to RGB
                center_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                # append rgb
                images.append(center_image_rgb)
                center_angle = float(batch_sample[3])
                angles.append(center_angle)
                # append flipped
                images.append(cv2.flip(center_image_rgb, 1))
                angles.append(-center_angle)
                
                # left image
                name = '/opt/carnd_p3/data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                # convert to RGB
                left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                # append RGB
                images.append(left_image_rgb)
                left_angle = float(batch_sample[3]) + 0.1
                angles.append(left_angle)
                # append flipped
                images.append(cv2.flip(left_image_rgb, 1))
                angles.append(-left_angle)
                
                # right image
                name = '/opt/carnd_p3/data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                # convert to RGB
                right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                # append RGB
                images.append(right_image_rgb)
                right_angle = float(batch_sample[3]) - 0.1
                angles.append(right_angle)
                # append flipped
                images.append(cv2.flip(right_image_rgb, 1))
                angles.append(-right_angle)
                
            # X_train and y_train, Convert to numpy array
            X_train = np.array(images)
            y_train = np.array(angles)
            # shuffle
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# import required methods
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Model based on NVIDIA paper: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
# normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
# cropping2D layer: 70 row pixels from the top of the image, 25 row pixels from the bottom of the image
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Applying 24 filter of sizes (5,5) of strides of 2 with relu activation
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
# Applying 36 filter of sizes (5,5) of strides of 2 with relu activation
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
# Applying 48 filter of sizes (5,5) of strides of 2 with relu activation
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
# Applying 64 filter of sizes (3,3) of strides of 1 with relu activation
model.add(Conv2D(64, (3,3), activation='relu'))
# Applying 24 filter of sizes (3,3) of strides of 1 with relu activation
model.add(Conv2D(64, (3,3), activation='relu'))

# dropout
model.add(Dropout(0.5))

# flatten and dense
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# using adam optimizer and mse loss function
model.compile(loss='mse', optimizer='adam')

# training the model, 32 is the batch_size
model.fit_generator(train_generator, steps_per_epoch= len(train_samples) * 10 // 32,
validation_data=validation_generator, validation_steps=len(validation_samples) // 32, epochs=5, verbose = 1)

# save the model
model.save('model.h5')