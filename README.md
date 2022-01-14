# Network Traffic Prediction
## Problem Statement 
Based on historical data of 10 different ports, the network traffic needs to be predicted.
There are four categories of Network-0,1,2 and 3. Three being the busiest and zero means normal network traffic
# Approach
## LSTM
The last 10 mins traffic is taken as timestep and dependent variable as Network traffic category.
## Other ML Algorithms 
10 features are created out of last ten min traffic status of different the port- one features for each min.
After data cleaning, PCA is performed to select the best features.
Different ML Algorithms like Random Forest, ANN, Decision tress are trained and tested to select the best model.

##Final Results
LSTM gave the best result with 98% 
Accuracy of  Neural Network Classifier on test set: 0.98

              precision    recall  f1-score   support

         0.0       0.98      1.00      0.99      1485
         1.0       0.97      0.98      0.97      1532
         2.0       0.99      0.94      0.96      1511
         3.0       0.98      0.99      0.98      1472

    accuracy                           0.98      6000
   macro avg       0.98      0.98      0.98      6000
weighted avg       0.98      0.98      0.98      6000

