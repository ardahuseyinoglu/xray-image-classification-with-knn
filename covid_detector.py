# IMPORT LIBRARIES 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

import time
from timeit import default_timer as timer

get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras.preprocessing import image
from keras.layers import merge, Input

import cv2 
import os
from tqdm import tqdm 
from zipfile import ZipFile
from PIL import Image


#-------------------------------------------------------------------------------------------------


# LOADING TRAINING DATASET

def load_class_data(class_dir_path, class_dir_name, X, Y):
    counter = 0
    for img_name in tqdm(os.listdir(class_dir_path), ascii=True, desc=class_dir_name):
        img_path = os.path.join(class_dir_path,img_name)
        #print(counter, img_path)
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256,256))
        X.append(np.array(img))
        Y.append(class_dir_name)
        counter += 1
    return counter

        
def load_data(path):
    X = []
    Y = []
    counts_class_samples = []
    for class_dir_name in os.listdir(path):
        class_dir_path = os.path.join(path,class_dir_name)
        number_of_sample = load_class_data(class_dir_path, class_dir_name, X, Y)
        counts_class_samples.append(number_of_sample)
    return X, Y, counts_class_samples


classes = os.listdir("train")
print(classes)

X, Y, number_of_samples = load_data(path="train")

# Label Encoding for Y values
le = LabelEncoder()
Y_ = le.fit_transform(Y)

#---------------------------------

# Sample Distribution on Training Data
sns.barplot(x=classes, y=number_of_samples, palette="pastel")

# Random Images from Dataset
fig,ax=plt.subplots(3,3)
fig.set_size_inches(10,10)

ax[0,0].imshow(X[10])
ax[0,0].set_title(Y[10])
ax[0,1].imshow(X[1000])
ax[0,1].set_title(Y[1000])
ax[0,2].imshow(X[2200])
ax[0,2].set_title(Y[2200])
ax[1,0].imshow(X[30])
ax[1,0].set_title(Y[30])
ax[1,1].imshow(X[1500])
ax[1,1].set_title(Y[1500])
ax[1,2].imshow(X[2500])
ax[1,2].set_title(Y[2500])
ax[2,0].imshow(X[100])
ax[2,0].set_title(Y[100])
ax[2,1].imshow(X[1700])
ax[2,1].set_title(Y[1700])
ax[2,2].imshow(X[2750])
ax[2,2].set_title(Y[2750])

plt.tight_layout()


#----------------------------------------------------------------------------------------------------------------


# FUNCTIONS FOR FEATURE EXTRACTION METHODS


# Canny Edge Detection
def Canny_edge(img):
    canny_edges = cv2.Canny(img,100,200)
    return canny_edges

# show a sample image
plt.imshow(Canny_edge(X[1000]))
plt.title("2-D image after implementing canny edge detection")

#---------------------------------

# Gabor Filtering
# Grayscale
def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray

# Gabor Filter
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get half size
    d = K_size // 2

    # prepare kernel
    gabor = np.zeros((K_size, K_size), dtype=np.float32)

    # each value
    for y in range(K_size):
        for x in range(K_size):
            # distance from center
            px = x - d
            py = y - d

            # degree -> radian
            theta = angle / 180. * np.pi

            # get kernel x
            _x = np.cos(theta) * px + np.sin(theta) * py

            # get kernel y
            _y = -np.sin(theta) * px + np.cos(theta) * py

            # fill kernel
            gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)

    # kernel normalization
    gabor /= np.sum(np.abs(gabor))

    return gabor

# Use Gabor filter to act on the image
def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get shape
    H, W = gray.shape

    # padding
    gray = np.pad(gray, (K_size//2, K_size//2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)
        
    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y : y + K_size, x : x + K_size] * gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out

# Use 6 Gabor filters with different angles to perform feature extraction on the image
def Gabor_process(img):
    # get shape
    H, W, _ = img.shape

    # gray scale
    gray = BGR2GRAY(img).astype(np.float32)

    # define angle
    #As = [0, 45, 90, 135]
    As = [0,30,60,90,120,150]

    # prepare pyplot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        _out = Gabor_filtering(gray, K_size=9, Sigma=1.5, Gamma=1.2, Lambda=1, angle=A)

        # add gabor filtered image
        out += _out

    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out

# show a sample image
plt.imshow(Gabor_process(X[1000]))
plt.title("2-D image after implementing gabor filtering")

#-----------------------------------------

# Tiny & Gray Scaled Image
def convert_tiny_grayscale(img,dim):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, dsize=(dim, dim), interpolation=cv2.INTER_CUBIC)
    return img

# show a sample image
plt.imshow(convert_tiny_grayscale(X[1000], dim=64))
plt.title("2-D image after converting image to tiny(64x64) and gray scaled")

#-----------------------------------------

# HOG Features
def convert_hog(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=True)
    return fd, hog_image

# show a sample image
plt.imshow(convert_hog(X[2950])[1])
plt.title("HOG features of image")



#--------------------------------------------------------------------------------------------------------


# FEATURE EXTRACTION (&saving feature vectors locally)


# Gabor Filtering
# apply gabor filtering to each image and flatten 2-d image to 1-d and store as feature vector.
for i in tqdm(range(len(X))):
    processed_img = Gabor_process(X[i])
    processed_img = cv2.resize(processed_img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC) #optional
    img_feature = processed_img.flatten()
    X[i] = img_feature

X = np.array(X)
# to save feature np.array to a csv.file:
np.savez_compressed('gabor_imgSize64x64_features_of_all_training_imgs.npz', X)

#------------------------------------------

# Canny Edge
# apply canny edge detection to each image and flatten 2-d image to 1-d and store as feature vector.
for i in tqdm(range(len(X))):
    processed_img = Canny_edge(X[i])
    img_feature = processed_img.flatten()
    X[i] = img_feature
    
X = np.array(X)
# to save feature np.array to a csv.file:
np.savez_compressed('canny_edge_features_of_all_training_imgs.npz', X)

#------------------------------------------

# Tiny Images
for i in tqdm(range(len(X))):
    processed_img = convert_tiny_grayscale(X[i], dim=32)
    img_feature = processed_img.flatten()
    X[i] = img_feature
    
X = np.array(X)
# to save feature np.array to a csv.file:
np.savez_compressed('32_tiny_features_of_all_training_imgs.npz', X)

#------------------------------------------

# Deep Features (Feature extraction using CNN models)

# VGG-16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

new_input = Input(shape=(256, 256, 3))
model = VGG16(include_top=False, weights='imagenet', input_tensor=new_input, pooling='avg')
model.summary()

for i in tqdm(range(len(X))):
    img_data_expanded = np.expand_dims(X[i], axis=0)
    image_processed = preprocess_input(img_data_expanded)
    vgg16_feature = model.predict(image_processed)
    vgg16_feature = vgg16_feature.flatten()
    X[i] = vgg16_feature

X = np.array(X)
# to save feature np.array to a csv.file:
np.savez_compressed('vgg16_pool_avg_features_of_all_training_imgs.npz', X)



# VGG-19
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

# specify the input image dimensions
new_input = Input(shape=(256, 256, 3))
# create the model
model = VGG19(include_top=False, weights='imagenet', input_tensor=new_input, pooling='avg')
model.summary()

for i in tqdm(range(len(X))):
    img_data_expanded = np.expand_dims(X[i], axis=0)
    image_processed = preprocess_input(img_data_expanded)
    vgg19_feature = model.predict(image_processed)
    vgg19_feature = vgg19_feature.flatten()
    X[i] = vgg19_feature

X = np.array(X)
# to save feature np.array to a csv.file:
np.savez_compressed('vgg19_pool_avg_features_of_all_training_imgs.npz', X)



# ResNet-50
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input

# specify the input image dimensions
new_input = Input(shape=(256, 256, 3))
# create the model
model = ResNet50(include_top=False, weights='imagenet', input_tensor=new_input, pooling='avg')
model.summary()

for i in tqdm(range(len(X))):
    img_data_expanded = np.expand_dims(X[i], axis=0)
    image_processed = preprocess_input(img_data_expanded)
    resnet50_feature = model.predict(image_processed)
    resnet50_feature = resnet50_feature.flatten()
    X[i] = resnet50_feature

X = np.array(X)
# to save feature np.array to a csv.file:
np.savez_compressed('resnet50_pool_avg_features_of_all_training_imgs.npz', X)



# InceptionV3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

# specify the input image dimensions
new_input = Input(shape=(256, 256, 3))
# create the model
model = InceptionV3(include_top=False, weights='imagenet', input_tensor=new_input, pooling='avg')
model.summary()

for i in tqdm(range(len(X))):
    img_data_expanded = np.expand_dims(X[i], axis=0)
    image_processed = preprocess_input(img_data_expanded)
    inception_v3_feature = model.predict(image_processed)
    inception_v3_feature = inception_v3_feature.flatten()
    X[i] = inception_v3_feature

X = np.array(X)
# to save feature np.array to a csv.file:
np.savez_compressed('inception_v3_pool_avg_features_of_all_training_imgs.npz', X)

#------------------------------------------

# HOG Features with/without equalizeHist
from skimage.feature import hog
from skimage import data, exposure

for i in tqdm(range(len(X))):
    img_gray = cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY)
    #img_equalizeHist = cv2.equalizeHist(src)  # optional (equalizeHist)
    fd, hog_image = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=True)
    img_feature = fd.flatten()
    X[i] = img_feature
    
X = np.array(X)
# to save feature np.array to a csv.file:
np.savez_compressed('hog_features_of_all_training_imgs.npz', X)

#------------------------------------------

# to select a feature extractor that you want
def get_feature_vectors(feature_extractor_name, tiny_size=32):
    if feature_extractor_name == "gabor":
        # to load feature np.array back to variable X:
        X_ = np.load('gabor_features_of_all_training_imgs.npz')
        X_ = X_['arr_0']
        X_ = X_ / 255
    
    elif feature_extractor_name == "gabor_64x64":
        X_ = np.load('gabor_imgSize64x64_features_of_all_training_imgs.npz')
        X_ = X_['arr_0']
        X_ = X_ / 255
    
    elif feature_extractor_name == "canny":
        X_ = np.load('canny_edge_features_of_all_training_imgs.npz')
        X_ = X_['arr_0']
        X_ = X_ / 255
    
    elif feature_extractor_name == "tiny":
        addr = str(tiny_size) + '_' + 'tiny_features_of_all_training_imgs.npz'
        X_ = np.load(addr)
        X_ = X_['arr_0']
        X_ = X_ / 255
        
    elif feature_extractor_name == "tiny32":
        addr = str(32) + '_' + 'tiny_features_of_all_training_imgs.npz'
        X_ = np.load(addr)
        X_ = X_['arr_0']
        X_ = X_ / 255
        
    elif feature_extractor_name == "tiny16":
        addr = str(16) + '_' + 'tiny_features_of_all_training_imgs.npz'
        X_ = np.load(addr)
        X_ = X_['arr_0']
        X_ = X_ / 255
        
    elif feature_extractor_name == "tiny64":
        addr = str(64) + '_' + 'tiny_features_of_all_training_imgs.npz'
        X_ = np.load(addr)
        X_ = X_['arr_0']
        X_ = X_ / 255
    
    elif feature_extractor_name == "vgg16_avg_pool":
        X_ = np.load('vgg16_pool_avg_features_of_all_training_imgs.npz')
        X_ = X_['arr_0']
        
    elif feature_extractor_name == "vgg16":
        X_ = np.load('vgg16_features_of_all_training_imgs.npz')
        X_ = X_['arr_0']
    
    elif feature_extractor_name == "vgg19_avg_pool":
        X_ = np.load('vgg19_pool_avg_features_of_all_training_imgs.npz')
        X_ = X_['arr_0']
        
    elif feature_extractor_name == "resnet50_avg_pool":
        X_ = np.load('resnet50_pool_avg_features_of_all_training_imgs.npz')
        X_ = X_['arr_0']
    
    elif feature_extractor_name == "inception_v3_avg_pool":
        X_ = np.load('inception_v3_pool_avg_features_of_all_training_imgs.npz')
        X_ = X_['arr_0']
        
    elif feature_extractor_name == "equalizeHisted_hog":
        X_ = np.load('equalizeHisted_hog_features_of_all_training_imgs.npz')
        X_ = X_['arr_0']
        
    elif feature_extractor_name == "hog":
        X_ = np.load('hog_features_of_all_training_imgs.npz')
        X_ = X_['arr_0']
        
    elif feature_extractor_name == "hog_concat_resnet50":
        X_resnet50 = np.load('resnet50_pool_avg_features_of_all_training_imgs.npz')
        X_resnet50 = X_resnet50['arr_0']
        X_hog = np.load('hog_features_of_all_training_imgs.npz')
        X_hog = X_hog['arr_0']
        X_ = np.concatenate((X_hog,X_resnet50), axis=1)
    
    elif feature_extractor_name == "hog_concat_vgg19":
        X_vgg19 = np.load('vgg19_pool_avg_features_of_all_training_imgs.npz')
        X_vgg19 = X_vgg19['arr_0']
        X_hog = np.load('hog_features_of_all_training_imgs.npz')
        X_hog = X_hog['arr_0']
        X_ = np.concatenate((X_hog,X_vgg19), axis=1)
        
    elif feature_extractor_name == "vgg16_concat_vgg19":
        X_vgg16 = np.load('vgg16_pool_avg_features_of_all_training_imgs.npz')
        X_vgg16 = X_vgg16['arr_0']
        X_vgg19 = np.load('vgg19_pool_avg_features_of_all_training_imgs.npz')
        X_vgg19 = X_vgg19['arr_0']
        X_ = np.concatenate((X_vgg16,X_vgg19), axis=1)
        
    elif feature_extractor_name == "vgg16_concat_hog_concat_resnet50":
        X_resnet50 = np.load('resnet50_pool_avg_features_of_all_training_imgs.npz')
        X_resnet50 = X_resnet50['arr_0']
        X_hog = np.load('hog_features_of_all_training_imgs.npz')
        X_hog = X_hog['arr_0']
        X_vgg16 = np.load('vgg16_pool_avg_features_of_all_training_imgs.npz')
        X_vgg16 = X_vgg16['arr_0']
        X_ = np.concatenate((X_vgg16,X_hog,X_resnet50), axis=1)
    
    elif feature_extractor_name == "gabor64x64_concat_resnet50":
        X_resnet50 = np.load('resnet50_pool_avg_features_of_all_training_imgs.npz')
        X_resnet50 = X_resnet50['arr_0']
        X_gabor64x64 = np.load('gabor_imgSize64x64_features_of_all_training_imgs.npz')
        X_gabor64x64 = X_gabor64x64['arr_0']
        X_ = np.concatenate((X_gabor64x64,X_resnet50), axis=1)
    
    elif feature_extractor_name == "hog_gabor64x64_concat_resnet50":
        X_resnet50 = np.load('resnet50_pool_avg_features_of_all_training_imgs.npz')
        X_resnet50 = X_resnet50['arr_0']
        X_gabor64x64 = np.load('gabor_imgSize64x64_features_of_all_training_imgs.npz')
        X_gabor64x64 = X_gabor64x64['arr_0']
        X_hog = np.load('hog_features_of_all_training_imgs.npz')
        X_hog = X_hog['arr_0']
        X_ = np.concatenate((X_hog,X_gabor64x64,X_resnet50), axis=1)
    
    return X_

	
#-----------------------------------------------------------------------------------------------------


# MODEL

def L1_dist(X_train, X):
    distances = np.sum(np.abs(X_train - X), axis=1)
    return distances


def L2_dist(X_train, X):
    distances = np.sqrt(np.sum(np.square(X_train - X), axis=1))
    return distances


def gaussian_kernel(dist, kernel_width):
    weight = np.exp(-(np.square(dist)/np.square(kernel_width)))
    return weight
	
	
def get_accuracy(preds, y):
    accuracy = np.sum((preds - y) == 0) / len(preds)
    return accuracy*100


# k-fold cross validation
class KFold_CrossValidation:
    
    def __init__(self, number_of_fold, shuffle=True):
        self.number_of_fold = number_of_fold
        self.shuffle = shuffle
    
    def split_to_folds(self,X):
        number_of_sample = X.shape[0]
        indices = np.arange(number_of_sample)
        
        if self.shuffle == True:    
            np.random.shuffle(indices)
            indices = indices.tolist()
        
        # get the number of samples in each fold
        number_of_samples_in_fold = int(np.ceil(number_of_sample/self.number_of_fold))
        
        # get an array of the starting indices of each fold in a shuffled array
        starting_indices = [number_of_samples_in_fold*i for i in range(self.number_of_fold)]
        
        folds = []
        for i in range(self.number_of_fold):
            
            # split the shuffled array and create each fold 
            if i != self.number_of_fold - 1:
                fold = indices[starting_indices[i]:starting_indices[i+1]]
            else:  #if it is the last fold
                fold = indices[starting_indices[i]:]
                
            # store each created fold in 'folds'
            folds.append(fold)
            
        splitted_train_valid_sets = []
        for i in range(len(folds)):
            valid_set = folds[i]
            train_set = []
            
            # concatenate folds except the fold choosen for validation set
            for j in range(len(folds)):
                if j == i:
                    continue
                train_set += folds[j]
                
            # create validation-training set pairs
            splitted_train_valid_sets.append([valid_set,train_set])
        
        return splitted_train_valid_sets


		
class KNearestNeighbor:
    def __init__(self):
        pass
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X, k, dist_metric, weighted, kernel_width):
        num_test_sample = X.shape[0]
        num_of_classes = 3
        y_pred = np.zeros(num_test_sample, dtype = int)
        
        for i in tqdm(range(num_test_sample), position=0, leave=True):
            
            #label_count is array of zeros used to store the votes (or sum of weights) for each class while comparing neighbours
            label_count = np.zeros(num_of_classes)
            
            if dist_metric == "L1": 
                distances = L1_dist(self.X_train, X[i])
            elif dist_metric == "L2": 
                distances = L2_dist(self.X_train, X[i])
            
            # since there are duplicates for some images in dataset (there is total 7 duplicates), 
            # we get k+1 neighbors in case of one of them is the same image and so the distance is 0.
            # If this is the case, we can use the (k+1)th closest neighbor instead of using same image.
            
            idx = np.argpartition(distances,k+1)
            min_ind = idx[:k]
            
            # replace the index of the duplicate image with (k+1)th closest neighbor
            for j in range(len(min_ind)):
                if distances[min_ind[j]] == 0.0:
                    min_ind[j] = idx[k]
                    break
            
            # weighted or not (options: inverse-distance or gaussian)
            if weighted == "non_weighted":
                for x in min_ind:
                    label_count[int(self.y_train[x])] += 1
            
            elif weighted == "inverse_distance":
                for x in min_ind:  
                    label_count[int(self.y_train[x])] += (1 / distances[x])
            
            elif weighted == "gaussian":
                for x in min_ind:
                    label_count[int(self.y_train[x])] += gaussian_kernel(distances[x], kernel_width)
            
            # select the index that has maximum value (index corresponds to class)
            y_pred[i] = np.argmax(label_count)
            
        return y_pred


		
def train_and_evaluate(X_, Y_, k_fold, k_Neighbors, distance_metric, weighted, kernel_width):
    
    # create k-fold object
    kfcv = KFold_CrossValidation(number_of_fold=k_fold, shuffle=True)
    
    # get all valid-training set pairs which is created by k-fold cross validation
    splitted_train_valid_sets = kfcv.split_to_folds(X_)
    fold_accuracy_scores = []
    all_preds = []
    valid_set_indices = []
    all_pred_time = []

    for fold_number, (valid_index, train_index) in enumerate(splitted_train_valid_sets):

        # split data into training set and validation set using k-fold cross validation.
        print("Fold " + str(fold_number+1) + ":", flush=True)
        X_train, X_valid = X_[train_index], X_[valid_index]
        Y_train, Y_valid = Y_[train_index], Y_[valid_index]
        print("Splitting is done for fold " + str(fold_number+1) + ".", flush=True)

        # train the model
        print("Training is proceeding...", flush=True)
        knn = KNearestNeighbor()
        knn.train(X_train, Y_train)
        print("Training is done.", flush=True)

        # get prediction (testing)
        print("Testing is proceeding...", flush=True)
        start = time.time()
        preds = knn.predict(X_valid, k_Neighbors, distance_metric, weighted, kernel_width)
        end = time.time()
        print("Testing is done.", flush=True)
        
        pred_time = end - start
        all_pred_time.append(pred_time)
        print("pred_time for the fold is " + str(pred_time), flush=True)
        
        # store all predictions and indices of validation samples
        all_preds.extend(preds)
        valid_set_indices.extend(valid_index)

        # get accuracy
        acc_score = get_accuracy(preds, Y_valid)
        fold_accuracy_scores.append(acc_score)
        print('\n Accuracy score for fold ' + str(fold_number+1) + ' : ' + str(acc_score) + '\n', flush=True)
        print("---------------------------------------------------------------", flush=True)
        
    total_pred_time = np.sum(all_pred_time)
    avg_pred_time_one_sample = np.sum(all_pred_time) / X_.shape[0]
    avg_acc_score = np.mean(fold_accuracy_scores)
    
    print('\n Average accuracy score: ' + str(avg_acc_score), flush = True)
    print('\n Total computation time for prediction: ' + str(total_pred_time), flush = True)
    print('\n Average computation time to predict a sample ' + str(avg_pred_time_one_sample),  flush=True)
    
    return avg_acc_score, all_preds, valid_set_indices, total_pred_time, avg_pred_time_one_sample


#----------------------------------------------------------------------------------------------------------


# MODEL SELECTION


# Feature Extraction Method Selection

feature_extraction_methods = ['vgg16_avg_pool', 
                              'vgg19_avg_pool', 
                              'resnet50_avg_pool',
                              'inception_v3_avg_pool',
                              'gabor', 
                              'canny', 
                              'tiny16',
                              'tiny32', 
                              'tiny64',
                              'hog',
                              'equalizeHisted_hog',
                              'hog_concat_resnet50', 
                              'hog_concat_vgg19', 
                              'gabor64x64_concat_resnet50', 
                              'hog_gabor64x64_concat_resnet50']


acc_results = []
total_pred_times = []
avg_pred_times_for_one_sample = []

for feature_extractor in feature_extraction_methods:
    # get a feature extractor method
    X_ = get_feature_vectors(feature_extractor)
    print(feature_extractor + '\n\n')

    avg_score, preds, indices_of_preds, total_pred_time, avg_pred_time_one_sample = train_and_evaluate(X_, Y_, 
                                                                                                       k_fold=5, 
                                                                                                       k_Neighbors=3, 
                                                                                                       distance_metric='L2', 
                                                                                                       weighted='non_weighted',
                                                                                                       kernel_width=None)
    acc_results.append(avg_score)
    total_pred_times.append(total_pred_time)
    avg_pred_times_for_one_sample.append(avg_pred_time_one_sample)
    print("---------------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------\n")

	

# The reason why I did some slicing ([5:]) is that I forgot to clear inside of the arrays which I used previously 
# and expreriment took 9 hours and I do not want to start again
total_prediction_times = np.array(pred_times[5:])
average_prediction_times_for_one_sample = np.array(avg_pred_times[5:])
names_of_feature_extraction_methods = feature_extraction_methods[:]	
accuracies_of_feature_extractors = np.array(acc_results[:])


# creating a dataframe that shows accuracy and computation time for prediction of each feature extraction method
df_feature_extraction = pd.DataFrame({'Method':names_of_feature_extraction_methods, 
                                      'Accuracy':accuracies_of_feature_extractors/100, 
                                      'Total prediction time (s)':total_prediction_times, 
                                      'Average prediction time for a sample (s)': (average_prediction_times_for_one_sample)})


# sort feature extraction methods by accuracy values in descending order
df_feature_extraction_sortedAccuracy = df_feature_extraction.sort_values('Accuracy', ascending=False)
display(df_feature_extraction_sortedAccuracy)

#------------------------------------------

# barplot to compare performance of different methods
plt1 = sns.barplot(data=df_feature_extraction_sortedAccuracy, 
                x='Method', 
                y='Accuracy',
                palette=sns.color_palette('mako', n_colors=15))

plt1.set_title("Accuracy comparisan between feature extraction methods")
plt1.set_xticklabels(plt1.get_xticklabels(), rotation=90)
plt1

#------------------------------------------

# scatterplot to observe computation time - accuracy relationship on each method
plt2 = sns.scatterplot(data=df_feature_extraction, 
                x='Average prediction time for a sample (s)', 
                y='Accuracy', 
                hue='Method', 
                palette=sns.color_palette('tab20', n_colors=15), 
                s=50, 
                alpha=0.8)

plt2.set_title("Accuracy vs Avg. Prediction time")
plt2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#------------------------------------------

# to observe the same scatter plot more closer to look at top left corner where better methods takes place
const_acc = df_feature_extraction['Accuracy'] >= 0.8
const_time = df_feature_extraction['Average prediction time for a sample (s)'] <= 0.5
df_better_extractors = df_feature_extraction[const_acc & const_time]

plt3 = sns.scatterplot(data=df_better_extractors, 
                x='Average prediction time for a sample (s)', 
                y='Accuracy', 
                hue='Method', 
                palette=sns.color_palette('tab20', n_colors=12), 
                s=100, 
                alpha=0.8)

plt3.set_title("Accuracy vs Avg. Prediction time (for top-left corner)")
plt3.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


#---------------------------------------------------------------------------------------------------


# HYPERPARAMETER TUNING

def tune_hyperparameter(hyperparameter_name, hyperparameters, feature_extraction_method):
    ''' gets corresponding hyperparameter values from the dictionary, cross validates these 
        hyperparameter values on the selected feature extraction method and plot the accuracy change 
        
        Args:
            hyperparameter_name (str): 'k_neighbor' | 'k_fold' | 'distance_metric' | 'kernel_width' | 'weighted'
            hyperparameters (dict): dictionary which store hyperparameter values for different hyperparameters
            feature_extraction_method (str): name of the method to be used to reload feature vectors
    '''
    
    values = hyperparameters[hyperparameter_name]
    X_ = get_feature_vectors(feature_extraction_method)
    acc_results = []
    
    if hyperparameter_name == 'k_neighbor':
        for value in values:
            print(value, "-------\n")
            avg_score, preds, indices_of_preds, total_pred_time, avg_pred_time_one_sample = train_and_evaluate(X_, Y_, 
                                                                                                               k_fold=5, 
                                                                                                               k_Neighbors=value, 
                                                                                                               distance_metric='L1', 
                                                                                                               weighted='gaussian',
                                                                                                               kernel_width=250)
            acc_results.append(avg_score)
            print("---------------------------------------------------------------------------------------------")
            print("---------------------------------------------------------------------------------------------\n")
            
            
    elif hyperparameter_name == 'k_fold':
        for value in values:
            print(value, "-------\n")
            avg_score, preds, indices_of_preds, total_pred_time, avg_pred_time_one_sample = train_and_evaluate(X_, Y_, 
                                                                                                               k_fold=value, 
                                                                                                               k_Neighbors=5, 
                                                                                                               distance_metric='L1', 
                                                                                                               weighted='gaussian',
                                                                                                               kernel_width=250)
            acc_results.append(avg_score)
            print("---------------------------------------------------------------------------------------------")
            print("---------------------------------------------------------------------------------------------\n")            
            
            
    elif hyperparameter_name == 'distance_metric':
        for value in values:
            print(value, "-------\n")
            avg_score, preds, indices_of_preds, total_pred_time, avg_pred_time_one_sample = train_and_evaluate(X_, Y_, 
                                                                                                               k_fold=5, 
                                                                                                               k_Neighbors=5, 
                                                                                                               distance_metric=value, 
                                                                                                               weighted='gaussian',
                                                                                                               kernel_width=250)
            acc_results.append(avg_score)
            print("---------------------------------------------------------------------------------------------")
            print("---------------------------------------------------------------------------------------------\n")            
            
            
    elif hyperparameter_name == 'kernel_width':
        for value in values:
            print(value, "-------\n")
            avg_score, preds, indices_of_preds, total_pred_time, avg_pred_time_one_sample = train_and_evaluate(X_, Y_, 
                                                                                                               k_fold=5, 
                                                                                                               k_Neighbors=5, 
                                                                                                               distance_metric='L1', 
                                                                                                               weighted='gaussian',
                                                                                                               kernel_width=value)
            acc_results.append(avg_score)
            print("---------------------------------------------------------------------------------------------")
            print("---------------------------------------------------------------------------------------------\n")            
            
            
    elif hyperparameter_name == 'weighted':
        for value in values:
            print(value, "-------\n")
            avg_score, preds, indices_of_preds, total_pred_time, avg_pred_time_one_sample = train_and_evaluate(X_, Y_, 
                                                                                                               k_fold=5, 
                                                                                                               k_Neighbors=5, 
                                                                                                               distance_metric='L1', 
                                                                                                               weighted=value,
                                                                                                               kernel_width=250)
            acc_results.append(avg_score)
            print("---------------------------------------------------------------------------------------------")
            print("---------------------------------------------------------------------------------------------\n")
    
    plt_knn = sns.lineplot(values, acc_results, linewidth=2, color='black')



# hyperparameter setup
hyperparameters = {'k_neighbor':[1,2,3,4,5,6,7,8,9,10,11,12], 
                   'k_fold':[4,5,6,7,8,9,10], 
                   'distance_metric':['L1', 'L2'], 
                   'kernel_width':[100, 250, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 
                                   1500, 1600, 1700, 1800, 1900, 2000, 2250, 2500, 2750, 3000], 
                   'weighted': ['non_weighted', 'inverse_distance', 'gaussian']}


# try different hyperparameters on selected method(resnet50)
tune_hyperparameter('distance_metric', hyperparameters, 'resnet50_avg_pool')


# Cross-validate the final model on selected hyperparameters & feature extractor
X_ = get_feature_vectors('resnet50_avg_pool')
avg_score, preds, indices_of_preds, total_pred_time, avg_pred_time_one_sample = train_and_evaluate(X_, Y_, 
                                                                                                   k_fold=5, 
                                                                                                   k_Neighbors=5, 
                                                                                                   distance_metric='L1', 
                                                                                                   weighted='gaussian', 
                                                                                                   kernel_width=250)


																								   

# View correctly classified or misclassified images
y_ground_truth = Y_[indices_of_preds]
predicted_correct = np.where(preds - y_ground_truth == 0)
predicted_wrong = np.where(preds - y_ground_truth != 0)
print(predicted_wrong)

def view_img_pred_groundTruth(index):
    ''' shows image and prints ground truth label & predictions for the given index
    
    Args:
        index (int): index value of the selected sample 
    '''
    
    print("Prediction label of given sample", classes[preds[index]])
    print("Ground truth label of given sample", classes[y_ground_truth[index]])
    print("sample no:", indices_of_preds[index])
    plt.imshow(X[index])

view_img_pred_groundTruth(841)


# Confusion Matrix
sns.heatmap(confusion_matrix(y_ground_truth, preds), 
            annot=True, 
            fmt="d", 
            cmap="YlGnBu",
            xticklabels=classes, 
            yticklabels=classes)


# Classification Report
print(classification_report(y_ground_truth, preds, target_names=classes))


#-----------------------------------------------------------------------------------------------


# TESTING


# Loading Test Images and Labels

def load_test_images(path):
    X = []    
    for img_name in tqdm(os.listdir(path), position=0, leave=True):
        img_path = os.path.join(path,img_name)
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256,256))
        X.append(np.array(img))
        
    return X

	
def load_test_labels(path):
    df = pd.read_csv("submission.csv")
    Y = list(df.Category)
    return Y
    
	
X_test = load_test_images(path="test")
Y_test = load_test_labels(path="submission.csv")

le = LabelEncoder()
Y_test_ =le.fit_transform(Y_test)

#------------------------------------------

# Feature Extraction for testset images(resnet50)

for i in tqdm(range(len(X_test)), position=0, leave=True):
    img_data_expanded = np.expand_dims(X_test[i], axis=0)
    image_processed = preprocess_input(img_data_expanded)
    resnet50_feature = model.predict(image_processed)
    resnet50_feature = resnet50_feature.flatten()
    X_test[i] = resnet50_feature

# backup features
X_test = np.array(X_test)
np.savez_compressed('resnet50_pool_avg_features_of_all_testing_imgs.npz', X_test)

# reload test set features
X_test_ = np.load('resnet50_pool_avg_features_of_all_testing_imgs.npz')
X_test_ = X_test_['arr_0']

# reload train set features and corresponding labels
X_train_ = get_feature_vectors('resnet50_avg_pool')
Y_train_ = Y_[:]

print(X_train_.shape)
print(Y_train_.shape)
print(X_test_.shape)
print(Y_test_.shape)

#------------------------------------------

# Test

def test_model(X_train_, Y_train_, X_test_, Y_test_):
    
    # train the model
    print("Training is proceeding...", flush=True)
    knn = KNearestNeighbor()
    knn.train(X_train_, Y_train_)
    print("Training is done.", flush=True)    

    # get prediction (testing)
    print("Testing is proceeding...", flush=True)
    preds = knn.predict(X_test_, k=7, dist_metric='L1', weighted='gaussian', kernel_width=250)
    print("Testing is done.", flush=True)
        
    # get accuracy
    acc_score = get_accuracy(preds, Y_test_)
    
    print("Accuracy : " + str(acc_score), flush=True)
        
    return acc_score, preds


# test model and get the accuracy score and all predictions for corresponding test data
accuracy_score, predictions = test_model(X_train_, Y_train_, X_test_, Y_test_)


# confusion_matrix
sns.heatmap(confusion_matrix(Y_test_, predictions), 
            annot=True, 
            fmt="d", 
            cmap="YlGnBu",
            xticklabels=classes, 
            yticklabels=classes)

			
# classification_report
print(classification_report(Y_test_, predictions, target_names=classes))


# show index of misclassified images
predicted_correct = np.where(predictions - Y_test_ == 0)
predicted_wrong = np.where(predictions - Y_test_ != 0)
print(predicted_wrong)


def view_img_pred_groundTruth(index):
    ''' shows image and prints ground truth label & predictions for the given index
    
    Args:
        index (int): index value of the selected sample 
    '''
    
    print("Prediction label of given sample", classes[predictions[index]])
    print("Ground truth label of given sample", classes[Y_test_[index]])
    plt.imshow(X_test[index])
    
    
view_img_pred_groundTruth(133)

#-------------------------------------------------------------------------------------------------------


# Create a dataframe fit the submission format (for Kaggle competition)

submission_df = pd.DataFrame({'Id':np.arange(1,778), 'Category':predictions})
submission_df = submission_df.astype({"Id": 'int64', "Category": str})

# change label encoded values to real labels
submission_df['Category'] = submission_df['Category'].str.replace('0','COVID')
submission_df['Category'] = submission_df['Category'].str.replace('1','NORMAL')
submission_df['Category'] = submission_df['Category'].str.replace('2','VIRAL')

# save dataframe as csv file
submission_df.to_csv('submission2_ah.csv', index=False)

