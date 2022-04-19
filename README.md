Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

# AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

# Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Moodle-Code Runner
Algorithm
Use the standard libraries in python for Gradient Design.
Set variables for assigning dataset values.
Import LinearRegression from the sklearn.
Assign the points for representing the graph.
Predict the regression for marks by using the representation of graph.
Compare the graphs and hence we obtain the LinearRegression for the given datas.

#Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: sarveshvaran p
RegisterNumber: 212221230090  
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("student_scores.csv")
dataset.head()

X = dataset.iloc[:,:-1].values
X
y = dataset.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
y_pred
y_test
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()

# Output:
![sarvesh1](https://user-images.githubusercontent.com/94881923/164035686-e4f0c9a9-73ae-4fdb-bdb8-f3291d607adb.png)
![sarvesh2](https://user-images.githubusercontent.com/94881923/164035730-54cc83bd-0b2b-455a-86ff-2cfb3d4718af.png)

# Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.


