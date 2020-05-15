# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[bar_chart_training_set]: ./output_images/bar_chart_training_set.png
"Distribution of training samples per label"
[labels_with_examples]: ./output_images/labels_with_examples.png "Labels and example images"
[grayscale]: ./output_images/grayscale.jpg "Grayscaling"
[traffic_signs_orig]: ./output_images/traffic_signs_orig.png "Traffic Signs"
[traffic_signs_prediction]: ./output_images/traffic_signs_prediction.png "Traffic Signs Prediction"
[prediction_probabilities_with_barcharts]: ./output_images/prediction_probabilities_with_barcharts.png "Traffic Sign Prediction with Bar Charts"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is （32，32，3）.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

The following figure shows one example image for each label in the training set.

![alt text][labels_with_examples]

Here is an exploratory visualization of the data set. It is a bar chart showing how many samples are contained in the training set per label.

![alt text][bar_chart_training_set]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because several images in the training were pretty dark and contained only little color und the grayscaling reduces the amount of features and thus reduces execution time.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][grayscale]

As a last step, I normalized the image data to obtain a mean zero and equal variance, formular: (pixel - 128)/ 128


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input 				| 32x32x1 gray scale image						| 
| Convolution 5x5 		| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					| 												|
| Max pooling			| 2x2 stride,  outputs 5x5x16 					|
| Flatten 				| outputs 400									|
| **Dropout** 			|												|
| Fully connected  		| outputs 120 									|
| RELU 					|												|
| **Dropout** 			|												|
| Fully connected  		| outputs 84 									|
| RELU 					|												|
| **Dropout** 			|												|
| Fully connected  		| outputs 43 									|
| Softmax  				|  												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer and the following hyperparameters:
* batch size: 128
* number of epochs: 100
* learning rate: 0.001
* Variables were initialized using the truncated normal distribution with mu = 0.0 and sigma = 0.1
* keep probalbility of the dropout layer: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.3%
* validation set accuracy of 96.8% 
* test set accuracy of 94.9%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? LeNet.
* What were some problems with the initial architecture? The training accuracy was under 90%. 
* How was the architecture adjusted and why was it adjusted? 
  Adding the grayscaling preprocessing, normalization, reduced learning rate and increased number of       epochs, for overfitting, added dropout layer after relu of final fully connected layer.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][traffic_signs_orig]

The "right-of-way at the next intersection" sign might be difficult to classify because the triangular shape is similiar to several other signs in the training set (e.g. "General caution" or "Slippery Road"). Additionally, the "Stop" sign might be confused with the "No entry" sign because both signs have more ore less round shape and a pretty big red area.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| right-of-way ..  		| right-of-way .. 								|
| Priority Road			| Priority Road  								|
| Turn left ahead		| Turn left ahead 								|
| Wild animals crossing	| Wild animals crossing  						|
| No entry				| No entry										|
| Yield					| Yield											|



The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.9%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a right-of-way at the next intersection sign(probability of 0.99), and the image does contain a right-of-way at the next intersection sign. The top six soft max probabilities were

| Probability 			|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| .99 					| right-of-way .. 								| 
| .00					| Pedestrians 									|
| .00					| Double curve									|
| .00					| Children crossingld							|
| .00				    | Beware of ice/snow 							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

