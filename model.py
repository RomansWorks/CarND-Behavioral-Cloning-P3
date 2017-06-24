import os
import csv
import numpy as np

samples = []
with open('./recorded_drives/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Commented out attempt to smooth values in the driving log, since I used keyboard to steer the vehicle...
#angles = np.array([sample[3] for sample in samples], dtype='float64')
#print(angles[0:200])
#weights = [0.1,0.3,0.6,0.3,0.1]
#smoothed_angles = np.convolve(angles,np.array(weights)[::-1],'same')
#for idx, sample in enumerate(samples):
#    sample[3] = smoothed_angles[idx]
#print(smoothed_angles[0:200])



from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import sklearn
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.core import Dropout

# Note: this is arbitrary
SIDE_CAMERA_ANGLE_CORRECTION = 0.2
# Note: if you run out of memory, decrease the batch size
BATCH_SIZE=128

# This generator returns a new batch on each yield.
# It does go over all batches in a specific epoch (shuffle) from first to last, and then restarts by reshufling the training set
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
                center_image = mpimg.imread(center_image_name)
                images.append(center_image)
                angles.append(center_angle)
                
                # Process side images
                left_image_name = './recorded_drives/IMG/'+batch_sample[1].split('/')[-1]
                left_image = mpimg.imread(left_image_name)
                left_angle = center_angle + SIDE_CAMERA_ANGLE_CORRECTION
                images.append(left_image)
                angles.append(left_angle)

                right_image_name = './recorded_drives/IMG/'+batch_sample[2].split('/')[-1]
                right_image = mpimg.imread(right_image_name)
                right_angle = center_angle - SIDE_CAMERA_ANGLE_CORRECTION
                images.append(right_image)
                angles.append(right_angle)
            
            # Augment data by including flipped images. Good against tendency to steer left.
            # Was not eventually needed to solve the assignment.
            #            flipped_images = [np.fliplr(image) for image in images]
            #            flipped_angles = [-1*angle for angle in angles]
            #
            #            images.extend(flipped_images)
            #            angles.extend(flipped_angles)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Crop some pixels from the top and bottom, to ignore non-road features and speed up training
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.))
# Above is -1 to 1, but we can also do 0-1: x/255 - 0.5. Which is more resilient?
# Below is the NVidia architecture (https://arxiv.org/abs/1604.07316)
# I set the activations to what seemed right (relu except for output layer), since it doesn't appear in the article
model.add(Convolution2D(24,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))


# Compile and fit the model. Learning rate is adaptive (ADAM optimizer)
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_samples)/BATCH_SIZE,
                                     validation_data=validation_generator,
                                     validation_steps=len(validation_samples)/BATCH_SIZE,
                                     epochs=10,
                                     verbose=1)
# Save the model
model.save('model.h5')

# Print the training and validation loss over epochs
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
