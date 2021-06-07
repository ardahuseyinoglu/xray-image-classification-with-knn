# Detection of Covid-19 disease from image using K-NN

In this assignment, we implement the nearest neighbor algorithm and its weighted version to classify X-ray images so that we can detect covid or viral pneumonia diseases. We also gain experience by using and implementing k-fold cross validation when tuning hyperparameters and selecting best feature extraction method to validate our model. Using different feature extraction methods is the significant part of the assinment as well. We are encouraged to try different methods to achieve best performance. Some of methods used are Canny Edge Detection, Gabor Filtering, Histogram Oriented Gradients (HOG), Deep Features (VGG-16, VGG-19, ResNet-50, Inception-V3) and different combination of them. We also will observe the effect of different hyperparameter choices on the model.


<p align="center">
  <b>Some sample images from training set</b>
</p>

<p align="center">
  <img src="/report-images/samples.png">
</p>

<p align="center">
  <b>Training set sample distribution</b> <br>
 Covid:960, Normal:1073, Viral Pneumonia:1076
</p>

<p align="center">
  <img src="/report-images/class_dist.png">
</p>


<p style="font-family:Times New Roman; font-size:18px"> Image features extracted from deep learning models give more successful results in general. They are also pretty good to predict a sample image in a short time in comparison with the other non-deep extraction methods. The reason why is that they use feature vectors that have less number of dimensions to represent an image and these features are more representative than other methods based on image processing techniques (less number of feature means small scale matrix operations when calculating distance between data points). ResNet-50 architecture gives the best accuracy among the other deep learning models so I use it and extend it with other methods by concatenating them to achive better accuracy rates. </p> 

<p style="font-family:Times New Roman; font-size:18px"> I use also different methods other than deep learning models: edge detection with Canny Edge Detector, Gabor filtering and HOG features. Canny edge detector turns out poor results. There can be two reasons for that. An image applied canny edge detection returns an 2-d numpy array which contains only 1's or 0's according to if the corresponding pixel belongs to an edge or not. And number of 1's(edge pixel) in an X-ray image applied edge detection is not high so it makes the matrix sparse. It may turns out that our canny edge features is not representative enough. The second reason can be due to distance metric used (L1 and L2). Since the feature vector of images only contain 0 or 1, most of the distance can be same even though they are not same image. For example, we have 3 different test data and let their feature vectors be [0,1,1,0], [1,0,0,1], [1,0,1,0] and let feature vector of train data be [1,1,1,1]. Then, L1 or L2 distances between training data and for each test data will be same even though ther are not same images.</p>  

<p align="center">
  <b>An example image after applying some feature extraction methods</b>
</p>

<p align="center">
  <img src="/report-images/4lü.PNG">
</p>

<p style="font-family:Times New Roman; font-size:18px"> Using smaller images can be other alternative if we do not use deep learning models. Although gabor filtering gives better accuracy rate, using tiny images decreases prediction time significantly and does not require any preprocessing in comparison with gabor filtering. </p> 

<p style="font-family:Times New Roman; font-size:18px">  I also used HOG features of an image by using the idea of the paper <i>[1] COVID-19 Detection from Chest X-ray Images Using Feature Fusion and Deep Learning.</i> They use HOG features of X-ray images beside CNN features to predict Covid-19 images form X-rays. Using HOG features alone as feature extraction method in our model give tolerable results as well (but the result is not as good as gabor filtering). So I decided to use HOG features with ResNet-50 features by concatenating them. It turns out to give best accuracy results among other models. However I get only a small improvement in comparisan with ResNet-50 features. It also give better result than the method of concatenating gabor features and ResNet-50 feature. That is interesting since I get more accuracy rate (+ 10%) when I use gabor filtering alone in comparison with HOG features alone.</p>

<p style="font-family:Times New Roman; font-size:18px"> However prediction time of HOG+ResNet-50 is almost ten times more than ResNet-50 and their accuracy rates are almost same (0.965582 and 0.964619, respectively), I decided to use only ResNet-50 features on my model. Lastly, I use the technique of histogram equalization to images using the OpenCV equalizeHist function from the paper <i>[2] X-Ray Image based COVID-19 Detection using Pre-trained Deep Learning Models.</i>. However, it does not give good results as expected.</p>

<p align="center">
  <b>Accuracy rate and computation time comparisans between feature extraction methods</b>
</p>

<p align="center">
  <img src="/report-images/comp3.png">
</p>

<p align="center">
  <img src="/report-images/df.PNG">
</p>

<p align="center">
  <img src="/report-images/comp1.PNG">
</p>

<p style="font-family:Times New Roman; font-size:18px"> In our model, there are different type of hyperparameters to be tuned, which are number of neighbors(k), distance metric and gaussian kernel width if weighted K-NN is used. Also we need to decide which K-NN algorithm we will use: weighted or non-weighted. If we use weighted K-NN, then there are also two options for how to choose weight values for K-NN: inverse distance or gaussian kernel. I first choose the feature extraction method as ResNet-50 features in according to the results based on accuracy and prediction time. Then, I try different hyperparameter values for each hyperparameter. Best neighbor value(k) for model accuracy is 5. However, I also observe the effect of when the number of neighbors equal to k=3 and k=7 with other hyperparameters. We can also see that when the number of neighbors is even and when the number of neighbors become large, accuracy decreases (The model accuracy decreases by oscillating when the k becomes larger). Since there can be same number of votes in non-weighted K-NN, the model just selects the closest index(class) among the classes that have same number of votes. </p>

<p style="font-family:Times New Roman; font-size:18px"> From the figure below, we can also see that using weighted K-NN with gaussian kernel as weight option gives the best accuracy in comparison with inverse distance weighted K-NN and non-weighted K-NN. Adding a weight factor improves the performance of the mode. I also observed that the optimal values for kernel width is completely depends on the problem itself and feature extraction method selected, since each feature extractor returns different range of values for an image. Thus, too small value or too small values for kernel width turns out that there are no voting. Since we use $e^{(-(x-x_n)^2/b^2)}$ when calculating weights, the result will come close to $0.0$ or $1.0$ if kernel width $b$ is too large or too small. I choose the kernel width as 250 but I also tried other values with different hyperparameter setting. Lastly, we can observe that using L1 as distance metric gives better accuracy rates.</p> 

<p align="center">
  <b>Hyperparameter Tuning - Accuracy changes on different hyperparameter values</b>
</p>

<p align="center">
  <img src="/report-images/comp2.PNG">
</p>

 <p style="font-family:Times New Roman; font-size:18px"> From the confusion matrix below, we see that our model predict 'Covid' labeled images more accurately. Most of the false predictions is due to predicting 'Normal' labeled images as 'Viral Pneumonia' labeled images and vice versa. The reason for this misclassification may be that X-ray images have different characteristics for different classes. For example, 'Covid' labeled X-ray images has more blurred / foggy characteristics. Another reason for this may be that 'Covid' labeled images includes some measuring devices on the patient chest. Those conditions can differentiate the 'Covid' labeled images from other classes easily. In addition, we can observe from the figure below that misclassified images resemble each other very much. And it can make harder to classify these images.</p>
 
 <p align="center">
  <b>Confusion matrix and classification report for test set</b>
</p>

<p align="center">
  <img src="/report-images/conf.PNG">
</p>

<p align="center">
  <b>Some misclassified images</b>
</p>

<p align="center">
  <img src="/report-images/misclass.PNG">
</p>

#### References

<p style="font-family:Times New Roman; font-size:17px"> [1] N.-A-A.; Ahsan, M.; Based, M.A.; Haider, J.; Kowalski, M. COVID-19 Detection from Chest X-ray Images Using Feature Fusion and Deep Learning. Sensors 2021, 21,1480. https://doi.org/10.3390/s21041480 </p> 
<p style="font-family:Times New Roman; font-size:17px"> [2] Horry, M. J., Chakraborty, S., Paul, M., Ulhaq, A., Pradhan, B., Saha, M., & Shukla, N. (2020, April 21). X-Ray Image based COVID-19 Detection using Pre-trained Deep Learning Models. https://doi.org/10.31224/osf.io/wx89s </p> 


