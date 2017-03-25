# **Traffic Sign Recognition** 


The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

The project code is [here](https://github.com/kkweon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

[//]: # (Image References)

[raw]: ./examples/raw.png "Datset"
[distplot]: ./examples/distplot.png "Distribution Plot"
[before_process]: ./examples/before_process.png "Before PreProcessing"
[after_process]: ./examples/after_process.png "After PreProcessing"
[data_augmentation]: ./examples/after_data_augmented_2.png "Data Augmented"
[new_dist]: ./examples/after_data_augmented_distplot.png "Data Augmented DistPlot"

[test]: ./examples/test_images.png "test images"
[test-result]: ./examples/test_images_result.png "test images result"

[test1]: ./examples/test1.png 
[test1-result]: ./examples/test1-softmax.png "test1 image"

[test2]: ./examples/test2.png 
[test2-result]: ./examples/test2-softmax.png "test2 image"

[test3]: ./examples/test3.png 
[test3-result]: ./examples/test3-softmax.png "test3 image"

[test4]: ./examples/test4.png 
[test4-result]: ./examples/test4-softmax.png "test4 image"

[test5]: ./examples/test5.png 
[test5-result]: ./examples/test5-softmax.png "test5 image"

### Data Set Summary & Exploration

#### 1. Dataset Summary

* The size of training set is 34,799
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43 (classes)

My dataset looks like this

![dataset][raw]

#### 2. More visualization

The code for this step is contained in the third code cell of the Jupyter notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the label distribution of the dataset
It shows that some classes have more/less examples than others and it can be a problem because the model will favor to the larger size examples instead of learning correct features.

![Distribution][distplot]

### Design and Test a Model Architecture

#### 1. Preprocessing Step

The code for this step is contained in the fourth code cell of the Jupyter notebook.

As a first step, I decided to normalize the images because if you look at above examples, it seems some images are brighter or darker than other images.
It can be difficult for my network to detect features because every bright pixel means higher number.

Here is an example of a traffic sign image before and after 

![Before][before_process]
![After][after_process]

Notice the last image was almost unvisible and after the normalization, it's visible again! It looks much sharper(higher contrasts) as well.

#### 2. Image Augmentation

Initially, I was able to download the german traffic dataset, which was given as follows

|          | Number of exmples|
|----------|-----------------:|
| Train set| 34,799           |
| Valid set|  4,410           |
| Test set | 12,630           |

Also from above, my train set has a imbalanced class problem. It means for some label, there are fewer/more examples than others such that my network choose to make a decision based on the distribution instead of learning distinct features.

Hence, I decided to create more artificial images by augmenting images.

![Augment][data_augmentation]

After several steps, I found out that rotation and random zoom works good.
So I created artificial images from these images to match up the balance.

![New DistPlot][new_dist]

and my training set increased to 86,010 images!


#### 3. Network Architecture

It's defined under Model Architecture in the jupyter notebook.

After trying out multiple architecures like GoogleNet, VGGnet, and LeNet, I decided to use the vggnet-like architecture because of training time and the performance per training time.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Conv 3x3 -> Bathcnorm -> Relu     	| 1x1 stride, same padding, outputs 32x32x64 	|
| Conv 3x3 -> Bathcnorm -> Relu       | 1x1 stride, same padding, outputs 32x32x64    |
| Max pooling           | 2x2 stride,  outputs 16x16x64                 |
| Conv 3x3 -> Bathcnorm -> Relu       | 1x1 stride, same padding, outputs 16x16x128   |
| Conv 3x3 -> Bathcnorm -> Relu       | 1x1 stride, same padding, outputs 16x16x128   |
| Max pooling	      	| 2x2 stride,  outputs 8x8x128  				|
| Conv 3x3 -> Bathcnorm -> Relu       | 1x1 stride, same padding, outputs 8x8x256     |
| Conv 3x3 -> Bathcnorm -> Relu       | 1x1 stride, same padding, outputs 8x8x256     |
| Max pooling           | 2x2 stride,  outputs 4x4x265                  |
| Conv 3x3 -> Bathcnorm -> Relu       | 1x1 stride, same padding, outputs 4x4x512     |
| Conv 3x3 -> Bathcnorm -> Relu       | 1x1 stride, same padding, outputs 4x4x512     |
| Max pooling           | 2x2 stride,  outputs 2x2x512                  |
| Flatten               |             									|
| Fully connected		| dimension 1024								|
| Batch Normalziation   |                                               |
| Dropout               |                                               |
| Fully connected       | dimension 1024                                |
| Softmax				| probability                                   |
 

#### 4. Hyperparameters

The code for training the model is located under Train Step in the notebook.

To train the model, I used the following parameters:

|     Parameter     |  Value |                                                                                              Reason                                                                                             |
|:-----------------:|:------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     Optimizer     |  Adam  | Because of its usage of momentum and adjustment step, I can be a bit more careless in choosing a learning rate                                                                                  |
|     Batch Size    |   256  | Batch Size is known to work best when it's times of 2 like 128, 256, 512.  The too small batch will be too stochastic. The bigger batch will take longer times to finish iterations.            |
|       Epoch       |   500  | I just ran it overnight but programmed to save a model when the validation loss is the minimum. Therefore, it works like an early stopping                                                      |
| L2 Regularization | 0.0001 | Regularization works always good Above this value, the network will begin to underfit!                                                                                                          |
|      Dropout      |   0.5  | When training mode, dropout 0.5 is applied for fully connected layers. I skipped the convolution layers because convolution layers have already few parameters, thus the dropout is unnecessary |
|   Batchnorm   |  True  | Always use batchnorm because its parameters can be also learned such that if the network does not need the batchnorm it will not use!                                                              |

#### 5. Performance

The code for calculating the accuracy of the model is located in the ninth cell of the Jupyter notebook.

My final model results were:

| Dataset |   Loss  | Accuracy |
|:-------:|:-------:|:--------:|
|  Train  | 0.00000 |  100.00% |
|  Valid  | 0.03842 |   99.07% |
|   Test  | 0.15470 |   98.01% |

I tried out LeNet5, GoogleNet(Inception-v2) and VGGnet. I had to reduce parameters because I was not using the pre-trained values and it turns out the network is so big that it takes forever to train in my machine.

I eventually chose a mini version with VGGnet-like architectures because 

* Multiple of 3x3 filters works same as 5x5 filters because its receptive fields are the same
* Training time is quicker than GoogleNet
* It still performed better than simpler LeNet5.

In conclusion, I am satisfied with these results though more data augmentations may increase test set accuracy.

### Test a Model on New Images

#### 1. Test against 5 new images found on the web

Here are five German traffic signs that I found on the web:

![test-images][test]

Each images is a sign of:
* No parking
* Schoolzone
* Road work
* Stop
* Right-of-way at the next intersection

After resizing, images look very rough that in the second image, it's hard to recognize it as a schoolzone signs.


#### 2. Top 1 Model Prediction

The code for making predictions on my final model is located in the last section of the notebook.

Here are the results of the prediction:

![test-images-result][test-result]


The model was able to correctly guess last 3 of the 5 traffic signs, which gives an accuracy of 60%. 
Using LeNet, it was only correctly guess 1 of the 5 traffic signs.

#### 3. Top 5 Model Prediction

The code for making predictions on my final model is located in the 11th cell of the Jupyter notebook.

For the first image, the highest probability is "Beware of ice/snow"(0.5).

![test1][test1]  
![test1-result][test1-result]

For the second image, the highest probability is 'Beware of ice/snow' with low confidence (0.3). Again there was no correct label(schoolezone) in the top 5 list.
The possible reasons might be that both pictures have the blue hue which is common in training images with the 'beware of ice/snow' label.

![test2][test2]  
![test2-result][test2-result]


For the third image, the model correctly predicted the sign of 'road work' with the highest confidence (1.0). The LeNet model was not predicting this correctly.

![test3][test3]  
![test3-result][test3-result]

For the fourth image, the model correctly predicted the 'stop' sign with some confidence (0.9). The LeNet model was not predicting this correctly.

![test4][test4]  
![test4-result][test4-result]

For the fifth image, the model correctly predicted the 'right-of-way at the next intersection' sign with the highest confidence (1.0). 
The LeNet model was predicting this correctly as well.

![test5][test5]  
![test5-result][test5-result]
