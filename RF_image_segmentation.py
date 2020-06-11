4р# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:57:22 2020

@author: Korisnik
"""

import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
from operator import itemgetter
import glob
import os

# list with all images
images = [cv2.imread(file) for file in glob.glob("\\Users\\Korisnik\\Desktop\\2ndYear\\IIP\\projekat\\frames\\*.JPG")]

# folder path where we will save images after preprocessing
path = 'C:\\Users\\Korisnik\\Desktop\\2ndYear\\IIP\\projekat\\ramovi'

#preprocessing all images, we will cut outside the white frame
num = 1
for img in images:
    #using grabcut
    img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA )

    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (5,5,250,250) 
    cv2.grabCut(img, mask, rect, bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    #plt.imshow(img)
    #plt.colorbar()
    #plt.show()
    
    ###########################################################################
   
    # resize image
    #img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #im_copy = img.copy()
    #plt.imshow(img)
    #plt.colorbar()
    #plt.show()

    #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #thresh = cv2.threshold(gray,240,255,0)[1]
    #plt.imshow(thresh,cmap='gray')
    #plt.show()

    #npImg = np.asarray(thresh)
    #coordList = np.argwhere(npImg > 200)
    #numWhitePoints = len( coordList )
    #print(coordList)
    
    #array_of_tuples = map(tuple, coordList)
    #tuple_of_tuples = list(array_of_tuples)
    #print(tuple_of_tuples)
    
    #max_y = max(tuple_of_tuples, key = itemgetter(1))[1]
    #max_x = max(tuple_of_tuples, key = itemgetter(0))[0]
    #min_x = min(tuple_of_tuples, key = itemgetter(0))[0]
    #min_y = min(tuple_of_tuples, key = itemgetter(1))[1]
    
    #cv2.rectangle(img, (min_y, min_x), (max_y, max_x), (255,255,255),-1)
    #plt.imshow(img)
    
    #img1 = (im_copy - img)
    #save preprocessed image
    cv2.imwrite(os.path.join(path , 'JPG_'+str(num)+'.jpg'), img)
    #cv2.waitKey(0)
    num += 1
    #plt.imshow(img1)
    
#defining function for feature extraction 
def feature_extraction(img):
    df = pd.DataFrame()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Save RGB Image pixels as separate features
    r, b, g = cv2.split(img)
    r = r.reshape(-1)
    b = b.reshape(-1)
    g = g.reshape(-1)
    df['Red Channel'] = r
    df['Blue Channel'] = b
    df['Green Channel'] = g
    
    #Save gray scale image pixels into a data frame. This is our Feature #1.
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img2 = img1.reshape(-1)
    df['Gray Scale Image'] = img2

    #Generate Gabor features on Gray pixels
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []
    #for theta in range(2):   #Define number of thetas
    theta = 1 / 4. * np.pi
        #for sigma in (1, 2):  #Sigma with 1 and 3
    sigma = 1
            #for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
    lamda = 3*np.pi/4
    for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
            
                
        gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
        #print(gabor_label)
        ksize=9
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
        kernels.append(kernel)
        #Now filter the image and add values to a new column 
        fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
        filtered_img = fimg.reshape(-1)
        df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc
        print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
        num += 1  #Increment for gabor column label
                
########################################
#Gerate OTHER FEATURES from Gray pixels and add them to the data frame
                
    #CANNY EDGE
    edges = cv2.Canny(img1, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe

    from skimage.filters import roberts, sobel, scharr, prewitt

    #ROBERTS EDGE
    edge_roberts = roberts(img1)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1

    #SOBEL
    edge_sobel = sobel(img1)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

    #SCHARR
    edge_scharr = scharr(img1)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    #PREWITT
    edge_prewitt = prewitt(img1)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1

    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img1, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img1, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

    #MEDIAN with sigma=3
    median_img = nd.median_filter(img1, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    
    return df

#loading all images and extracting faetures from them
images = [cv2.imread(file) for file in glob.glob("C:\\Users\\Korisnik\\Desktop\\2ndYear\\IIP\\projekat\\ramovi\\*.JPG")]
df = pd.DataFrame()
for i in  range(63):
    data = feature_extraction(images[i])
    df = pd.concat([df, data])

#loading mask images    
import matplotlib.image as mpimg
labeled_images = [cv2.imread(file) for file in glob.glob("C:\\Users\\Korisnik\\Desktop\\2ndYear\\IIP\\projekat\\bwMasks\\*.PNG")]
labels_list = []

for i in  range(63):
    mask = labeled_images[i]
    mask = cv2.resize(mask, (256, 256), interpolation = cv2.INTER_AREA)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Transform mask to binary image
    mask01 = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] > 126:
                mask01[i,j] = 255
            else: 
                mask01[i,j] = 0
    
    mask1 = mask01.reshape(-1).astype(int).tolist()
    labels_list.extend(mask1)
df['Labels'] = labels_list

#Define the dependent variable that needs to be predicted (labels)
Y = df["Labels"].values

#Define the independent variables
X = df.drop(labels = ["Labels"], axis=1) 

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)


#Import the RF Classifier
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators = 100, max_features = np.log(2), random_state = 20, class_weight = {0:1,255:10})

# Train the model on training data
model.fit(X_train, y_train)


#First test prediction on the training data itself 
prediction_test_train = model.predict(X_train)

#Test prediction on testing data. 
prediction_test = model.predict(X_test)


from sklearn import metrics
#Print the prediction accuracy

#First check the accuracy on training data. This will be higher than test data prediction accuracy.
print ("Accuracy on training data = ", metrics.accuracy_score(y_train, prediction_test_train))
#Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

#feature importances
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)

import seaborn as sns
sns.barplot(x=feature_imp.values, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.savefig("importance1.png", dpi=100, bbox_inches='tight')
plt.show()



#Save the trained model as pickle string to disk for future use
import pickle
filename = "RF_Image_Segmentation"
pickle.dump(model, open(filename, 'wb'))

#f1 scores
from sklearn.metrics import f1_score
f1_macro_test = f1_score(y_test,prediction_test, average='macro')
f1_micro_test = f1_score(y_test,prediction_test, average='micro')
f1_weighted_test = f1_score(y_test,prediction_test, average='weighted')

print("F1 macro score is: ", f1_macro_test)
print("F1 micro score is: ", f1_micro_test)
print("F1 weighted score is: ", f1_weighted_test)

#To test the model on future datasets
loaded_model = pickle.load(open(filename, 'rb'))

###SVM model
from sklearn.svm import LinearSVC
model = LinearSVC(max_iter=100)  


##############
img = cv2.imread('C:\\Users\\Korisnik\\Desktop\\2ndYear\\IIP\\projekat\\vezbanje\\1_2_2.JPG')
img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA )


X = feature_extraction(img)

result = model.predict(X)

segmented = result.reshape((256,256))


plt.imshow(segmented, cmap ='gray')
    
#plt.imsave('segmented_RF_100_estim.jpg', segmented)

cv2.imwrite('C:\\Users\\Korisnik\\Desktop\\2ndYear\\IIP\\projekat\\vezbanje\\segmented1_RF_5000_estim.jpg', segmented) 

#accuracy depending on number of estimators(trees) in RF classifier 
num_trees = [50, 100, 200]
acc_list =[0.9525231236164275, 0.9525171631533711, 0.952518951292288]
plt.plot(num_trees, acc_list,'ro--')
plt.xlabel('Number of trees in RF')
plt.ylabel('Accuracy')
plt.title('Change of accuracu with number of trees')
plt.grid(True)
plt.savefig("trees_acc.png", dpi=100)
plt.show()

########## classification report
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(y_test, prediction_test, target_names=target_names))

########gridsearch to reduce overfitting
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier




rfc = RandomForestClassifier(n_estimators=50, random_state = 20 ) 

param_grid = { 
    'n_estimators': [200, 50],
    'max_features': [ 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)

print (CV_rfc.best_params_)


#contours

seg = cv2.imread('C:\\Users\\Korisnik\\Desktop\\2ndYear\\IIP\\projekat\\vezbanje\\segmented1_RF_5slika_estim.jpg') 
imgray = cv2.cvtColor(seg, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(seg, 127, 255, 0)
contours= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
print(contours)
print(defects)
img_contours = np.zeros(im.shape)
c = cv2.drawContours(img_contours, contours[1], -1, (0,255,0), 3)
plt.imshow(c)