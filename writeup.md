#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

Overfitting was indeed a problem. Here's example (partial) output of one of the more advanced models.
In this model, dropout was already employed, but only on one layer and with a low 0.3 dropout probability:
```
126/125 [==============================] - 69s - loss: 0.0376 - val_loss: 0.0327
Epoch 2/30
126/125 [==============================] - 63s - loss: 0.0296 - val_loss: 0.0306
Epoch 3/30
126/125 [==============================] - 63s - loss: 0.0284 - val_loss: 0.0300
Epoch 4/30
126/125 [==============================] - 63s - loss: 0.0272 - val_loss: 0.0292
Epoch 5/30
126/125 [==============================] - 63s - loss: 0.0264 - val_loss: 0.0284
Epoch 6/30
126/125 [==============================] - 63s - loss: 0.0251 - val_loss: 0.0280
Epoch 7/30
126/125 [==============================] - 63s - loss: 0.0234 - val_loss: 0.0265
Epoch 8/30
126/125 [==============================] - 63s - loss: 0.0221 - val_loss: 0.0262
Epoch 9/30
126/125 [==============================] - 63s - loss: 0.0220 - val_loss: 0.0263
Epoch 10/30
126/125 [==============================] - 64s - loss: 0.0202 - val_loss: 0.0249
Epoch 11/30
126/125 [==============================] - 63s - loss: 0.0189 - val_loss: 0.0238
Epoch 12/30
126/125 [==============================] - 64s - loss: 0.0167 - val_loss: 0.0242
Epoch 13/30
126/125 [==============================] - 62s - loss: 0.0152 - val_loss: 0.0227
Epoch 14/30
126/125 [==============================] - 63s - loss: 0.0138 - val_loss: 0.0232
Epoch 15/30
126/125 [==============================] - 63s - loss: 0.0130 - val_loss: 0.0236
Epoch 16/30
126/125 [==============================] - 63s - loss: 0.0118 - val_loss: 0.0223
Epoch 17/30
126/125 [==============================] - 63s - loss: 0.0109 - val_loss: 0.0238
Epoch 18/30
126/125 [==============================] - 62s - loss: 0.0104 - val_loss: 0.0223
Epoch 19/30
126/125 [==============================] - 63s - loss: 0.0099 - val_loss: 0.0232
Epoch 20/30
126/125 [==============================] - 62s - loss: 0.0095 - val_loss: 0.0221
Epoch 21/30
126/125 [==============================] - 63s - loss: 0.0108 - val_loss: 0.0244
Epoch 22/30
126/125 [==============================] - 63s - loss: 0.0104 - val_loss: 0.0223
Epoch 23/30
126/125 [==============================] - 63s - loss: 0.0090 - val_loss: 0.0219
Epoch 24/30
126/125 [==============================] - 63s - loss: 0.0078 - val_loss: 0.0221
Epoch 25/30
126/125 [==============================] - 63s - loss: 0.0073 - val_loss: 0.0221
Epoch 26/30
126/125 [==============================] - 62s - loss: 0.0068 - val_loss: 0.0217
Epoch 27/30
126/125 [==============================] - 63s - loss: 0.0064 - val_loss: 0.0228
Epoch 28/30
126/125 [==============================] - 64s - loss: 0.0063 - val_loss: 0.0223
Epoch 29/30
126/125 [==============================] - 62s - loss: 0.0060 - val_loss: 0.0220
Epoch 30/30
126/125 [==============================] - 62s - loss: 0.0057 - val_loss: 0.0218
```

I resolved this by adding three dropout layers, one after each of the first three densly connected layers.
The probabilities were then set to 0.5, 0.4, 0.3. 

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line _____).

####4. Appropriate training data

I collected about 5 laps of center lane driving in the regular direction. 
I then collected several recordings of recovery driving which started near the road edge, 
with the vehicle facing somewhat outwards of the lane center (as would happen in case of incorrect steering),
and ending in return to road center. 
I then repeated both of these processes while driving in the reverse direction of the track.
Eventually, I had to augment the training data by doing several center lane drives near the bridge, to
balance out multiple recovery drivings at the bridge.
I also had to collect additional reverse lane driving data to resolve a tendency to steer to the left.


###Model Architecture and Training Strategy

####1. Solution Design Approach

I used the general architecture from https://arxiv.org/abs/1604.07316
Since the activations are not shown there, I used the common RelU activations for hidden layers, 
and the smoother tanh activation for the ouput neuron.

As mentioned above, I had to augment the network with dropout layers to handle overfitting.

When feeding in the data, I first removed irrelevant parts on the top and buttom of the image, as well as normalized it (code from the lesson, adapted).

While I was testing the model by running the simulator, I encountered multiple errors made by the vehicle.

Most were resolved by adding training data and preventing overfitting.

Note that the code was tested and executed on a AWS P1 instance with TensorFlow 1.2.0 and Keras 2.0.2
Since the generator and Keras parameters follow the Keras 2 convention, the code will probably fail to load on older versions.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
