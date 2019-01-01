# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./examples/center_image.jpg "Center image"
[image3]: ./examples/center_image_flip.jpg "Center image flipped"
[image4]: ./examples/left_image.jpg "Left Image"
[image5]: ./examples/left_image_flip.jpg "Left image flipped"
[image6]: ./examples/right_image.jpg "Right Image"
[image7]: ./examples/right_image_flip.jpg "Right image flipped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on [NVIDIA paper](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The model uses 6 convolutional layers and 4 fully connected dense layers. The code can be found in lines: 93 to 115
The data is normalized in the model using a Keras lambda layer (code line 96). I have used RELU activation, and also added a dropout layer to reduce overfitting.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 111). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 121).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I started off with the sample data provided by Udacity and augmented the data to make better learning. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train the car to run through the drivable portion of the track surface.

My first step was to use a convolution neural network model similar to the [NVIDIA paper](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) I thought this model might be appropriate because this has been implented to a real car. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I could add dropout. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I augmented data to add flipped images. This made the model train better.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture can be found in lines 93-118 of model.py. It consists of normalization layer, followed by 6 convolutional neural network and 4 fully connected layers.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)
I have used the same architecture as [Nvidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) with same filter, but different size input. I have also added a dropout layer with 0.5 probability to reduce overfitting.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I have used the data provided by Udacity. Since model was not training well on initial dataset, I augmented the data by flipping each image vertically. Augmentation was done using a generator to speed up the training process. Code of generator function can be found in lines 28 to 81.

Here are examples of flipped image

**Center image** (Original and flipped) 

![alt text][image2] ![alt text][image3]

**Left image** (Original and flipped)

![alt text][image4] ![alt text][image5]

**Right image** (Original and flipped)

![alt text][image6] ![alt text][image7]

The dataset for training was split into training and validation. I set 20% of the data for validation. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by training. I used an adam optimizer so that manually training the learning rate wasn't necessary.