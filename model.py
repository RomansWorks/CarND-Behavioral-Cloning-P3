import os
import csv

samples = []
with open('./recorded_drives/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.core import Dropout

SIDE_CAMERA_ANGLE_CORRECTION = 0.2
BATCH_SIZE=128


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Read the actions taken by the drive in this image
                center_angle = float(batch_sample[3])
                throttle = float(batch_sample[4])
                brakes = float(batch_sample[5])
                
                # Process center image
                center_image_name = './recorded_drives/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_image_name)
                images.append(center_image)
                angles.append(center_angle)
                
                # Process side images
                left_image_name = './recorded_drives/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_image_name)
                left_angle = center_angle + SIDE_CAMERA_ANGLE_CORRECTION
                images.append(left_image)
                angles.append(left_angle)

                right_image_name = './recorded_drives/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_image_name)
                right_angle = center_angle - SIDE_CAMERA_ANGLE_CORRECTION
                images.append(right_image)
                angles.append(right_angle)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
# Consider also cropping=((50,20), (0,0))
model.add(Lambda(lambda x: x/127.5 - 1.))
# Above is -1 to 1, but we can also do 0-1: x/255 - 0.5. Which is more resilient?
model.add(Convolution2D(24,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))



model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_samples)/BATCH_SIZE,
                                     validation_data=validation_generator,
                                     validation_steps=len(validation_samples)/BATCH_SIZE,
                                     epochs=6,
                                     verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

print(history_object.history['loss'])
print(history_object.history['val_loss'])

#### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
