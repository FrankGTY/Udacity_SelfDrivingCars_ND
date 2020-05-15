# **Behavioral Cloning** 

## Writeup 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with 3x3 and 5x5 filter sizes all with RELU activations (model.py lines 63-73) 

There is a Lambda pre-processing layer to normalize the data followed by a 2D cropping layer to remove unhelpful pixel data from the input images (model.py lines 61-62). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 72). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 54-58). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 75).

I used 5 epochs for the model training since the improvement per epoch tapered off around the fourth or fifth epoch. Using a small batch size of 32 also helped with my model accuracy.

#### 4. Appropriate training data

The focus of my training data is to keep the vehicle in the center of the road. In situations where the vehicle veers close to the edge of the road I want it to understand how to react and recover. Since there are two different tracks for the vehicle to drive on, I eventually used data from both tracks.

After collecting training data I put together some additional functions to augment and preprocess the data prior to training. To ensure that the vehicle is properly equipped to react from its left side as well as its right side, I horizontally flipped all of my training images and multiplied the corresponding steering angles by -1 (model.py lines 39-40). This simulated more driving data as if the vehicle were performing the same correct turns in the opposite direction.
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find one that already works really well, and use that! In class we learned about several successful models when it comes to image recognition (AlexNet, GoogLeNet, etc). However the model that stuck out to me the most was the NVIDIA deep learning model since it was designed specifically for self-driving cars.

From there I followed the standard approach of splitting my data into training and validation data sets, preprocessing and augmenting that data, training the model and then testing its accuracy on the validation data.

To prevent overfitting, I modified the NVIDIA model by adding in one additional 50% dropout layer between the last two fully connected layers (model.py line 72). With this addition my training loss and validation became more consistent. 

#### 2. Final Model Architecture

The final model architecture (model.py lines 63-73) consisted of a convolution neural network with the following layers and layer sizes:
| Layer | Details |
| ------------------- | ------------------- |
| Input | Input shape of 160x320 with depth of 3 |
| Normalization | Centered around 0 between -0.5 and 0.5 |
| Cropping | Crop out top 70 and bottom 25 pixels |
| Convolutional | Depth of 24, 2×2 stride, 5×5 kernel, ReLu activation |
| Convolutional | Depth of 36, 2×2 stride, 5×5 kernel, ReLu activation |
| Convolutional | Depth of 48, 2×2 stride, 5×5 kernel, ReLu activation |
| Convolutional | Depth of 64, non-strided, 3×3 kernel, ReLu activation |
| Convolutional | Depth of 64, non-strided, 3×3 kernel, ReLu activation |
| Flatten | Size of 1164 |
| Fully Connected | Size of 100 |
| Fully Connected | Size of 50 |
| Fully Connected | Size of 10 |
| Dropout | 50% dropout rate |
| Output | Value representing the steering angle |

#### 3. Creation of the Training Set & Training Process


The training data was preprocessed by normalizing the data around 0 between -0.5 and 0.5 and then cropping the data to remove useless information at the top and bottom. Then the data was shuffled and split so that 20% of the data would be used for validation.

I used an Adam optimizer so I did not need to worry about setting a learning rate for the training process. The training loss improvement tapered off after the fourth or fifth epoch so I settled on 5 epochs for my training process.
