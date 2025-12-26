#Sonar ROCK v/s MINE detector:
'''
A submarine is going in an ocean and it has to detect inside the ocean whether any rocks or mines are present or not.
Sonar data is fed to our machine learning model to detech whether its a rock or a metal.
The sonar data will be given to us, which will be fed to the machine learning model to detech whether its a rock or a mine.
'''
#Adding the required libraries:

import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore #This library is used as the training data is first split and then fed into the machine learning model.
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

#Loading the dataset into pandas:

sonar_data=pd.read_csv('/Users/kushsharma/Downloads/archive 2/sonar.csv',header=None) #Path of the csv file is passed
#Header=None means that in the dataset file of sonar dataset there are no column names given to the files.

sonar_data.head() #This will print the first 5 rows of the dataset

#Finding out the number of rows and columns:

sonar_data.shape

#Defining our data, statistical methods of the data:

sonar_data.describe()

#Counting the number of mines and rocks:
 
sonar_data[60].value_counts()#60 is passed as the mines & rocks are at the 60th column.

sonar_data.groupby(60).mean() #This will calculate the mean of every column for the Mine and Rock.

#Seperating data and labels:
x=sonar_data.drop(columns=60,axis=1) #We are dropping the column 60 of the labels(rock and mine).
#While dropping the column the axis is set as 1.
y=sonar_data[60] #The dropped column is stored in "y"
print(x)
print(y)

#Training and Test Data:
'''
We are gong to split the data into train and test data so we have split it into x and y sets for train and test.
x and y are passed as arguments in train_test_split and test_size=0.1 means that we need 10% of the data, example if we have
200 data then 0.1% of 200 will be taken nto consideration.
Stratify=y means that data will be split into rock and mines and random_state=1 means that data is split into some specific way.

'''
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.1, stratify=y,random_state=1)

print(x.shape,x_train.shape,x_test.shape) 
'''
As can be seen in the output in "x.shape" we have 208,60 in x_train we have 187,60 and in x_test we have 21,60 data where 187 is the 
tran data and 21 is the test data.
'''
print(x_train)
print(y_train)

#Training the logistic regresson model with training data:

model=LogisticRegression()
model.fit(x_train,y_train)

#Accuracy on traning data:
'''
In this the x_train will be compared with the original data of y_train to find the accuracy.
'''
x_train_prediction=model.predict(x_train) #This wll predict the x_train model.
training_data_accuracy=accuracy_score(x_train_prediction,y_train) #The x_train model accuracy is compared with y_train.

print("Accuracy score of the train data is: ",training_data_accuracy*100) #we get almost 83 percent of accuracy.

#Accuracy on test data:

x_test_prediction=model.predict(x_test)
training_data_accuracy=accuracy_score(x_test_prediction,y_test)
print("Accuracy of the test data is: ",training_data_accuracy*100) #We get almost 76% of accuracy.

#Accuracy above 70% is considered a good accuracy for the model.

#Making a predictive system:
#From the set of sonar data given the values tll the column 59 are passed and our model will predict whether it is rock or mine.
input_data=(0.01,0.0171,0.0623,0.0205,0.0205,	0.0368,	0.1098,	0.1276,	0.0598,	0.1264,	0.0881,	0.1992,	0.0184,	0.2261,	0.1729,	0.2131,	0.0693,	0.2281,	0.406,	0.3973,	0.2741,	0.369,	0.5556,	0.4846,	0.314,	0.5334,	0.5256,	0.252,	0.209,	0.3559,	0.626,	0.734,	0.612,	0.3497,	0.3953,	0.3012,	0.5408,	0.8814,	0.9857,	0.9167,	0.6121,	0.5006,	0.321,	0.3202,	0.4295,	0.3654,	0.2655,	0.1576,	0.0681,	0.0294,	0.0241,	0.0121,	0.0036,	0.015,	0.0085,	0.0073,	0.005,0.0044,	0.004,	0.0117)

input_data_as_numpy_array=np.asarray(input_data)

#Reshaping our array as we are predicting it for one instance:
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction) 
if(prediction=='R'):
    print("It is a rock\n")
else:
 print("It is a mine\n")
