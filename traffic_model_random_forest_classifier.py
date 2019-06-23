########   Summary of  Pre-Processing / EDA   #######
## There is NO MISSING VALUE FOUND
## New Data Frame created with each minute traffic as one variable
## There is NO POTENTIAL OUTLIERS
### THERE IS NO CORRELATION FOUND AMONG 10 INDEPENDENT VARIABLES 
### PCA AND RANDOM FOREST RANKING are done, and all 10 features/variables looks signficant.
## Scaling is NOT REQUIRED as all attributes's data value is between 0 and 1 with similar variance

########   Summary of  model using ENSEMBLE method-RANDOM FOREST   ####

## Accuracy of  Random Forest classifier on test set: 0.99

## CONFUSION MATRIX
## col_0   0.0   1.0   2.0   3.0
## Y                            
## 0.0    1484     1     0     0
## 1.0       0  1531     1     0
## 2.0       0     0  1483    28
## 3.0       0     0     0  1472

## PRECISION AND RECALL

##              precision    recall  f1-score   support

##         0.0       1.00      1.00      1.00      1485
##         1.0       1.00      1.00      1.00      1532
##         2.0       1.00      0.98      0.99      1511
##         3.0       0.98      1.00      0.99      1472

## avg / total       0.99      0.99      0.99      6000


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


df_traffic = pd.read_csv('D:/NewPD/Data Science/TCL-Prob/traffic.csv')


##### Pre_processing sections starts #####

print(df_traffic.describe()) 

## All the variable value between 0 and 1 , NO NEED FOR SCALING

## Numnber of null values  
print("Total Number of missing values",df_traffic.isnull().sum())

#NO MISSING VALUE FOUND#####

###  Creating a  new dataframe with each minute of 10 min network traffic as one variable ##

x1=[]
x2=[]
x3=[]
x4=[]
x5=[]
x6=[]
x7=[]
x8=[]
x9=[]
x10=[]
y=[]
count=0
for i in df_traffic.sequence_no :
    if i == 0.0:
        x1.append(df_traffic['traffic'][count])
    elif i==1.0:
        x2.append(df_traffic['traffic'][count])
    elif i == 2.0:
        x3.append(df_traffic['traffic'][count])
    elif i==3.0:
        x4.append(df_traffic['traffic'][count])
    elif i==4.0:
        x5.append(df_traffic['traffic'][count])
    elif i == 5.0:
        x6.append(df_traffic['traffic'][count])
    elif i==6.0:
        x7.append(df_traffic['traffic'][count])
    elif i==7.0:
        x8.append(df_traffic['traffic'][count])
    elif i == 8.0:
        x9.append(df_traffic['traffic'][count])
    elif i==9.0:
        x10.append(df_traffic['traffic'][count])
        y.append(df_traffic['target'][count])
    
    count+=1     

df1 = pd.DataFrame({"X1": x1,"X2": x2,"X3": x3,"X4": x4,"X5": x5,"X6": x6,"X7": x7,"X8": x8,"X9": x9,"X10": x10, "Y": y})  

print(df1.describe()) 

## All the variable value between 0 and 1 , NO NEED FOR SCALING

Y=df1['Y'] 
features=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10']
X=df1[features]


###  Calculating correlation among variables ###
plt.matshow(X.corr())
plt.show()



#####  Plotting Box plot to get Outliers #####
data_to_plot=[df1['X1'],df1['X2'],df1['X3'],df1['X4'],df1['X5'],df1['X6'],df1['X7'],df1['X8'],df1['X9'],df1['X10']]

# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data_to_plot)

# Save the figure
fig.show()
fig.savefig('fig3_test.png', bbox_inches='tight')

## After checking Box plot , THERE IS NO POTENTIAL OULIERS FOUND ###

 #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42)


###  Feature selection using PCA ###

# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
########################################################



n_components = 10 #we have 10 features
covar_matrix = PCA(n_components) 
covar_matrix.fit(X_train)
variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios

var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')

plt.plot(var)
# from the plot, ALL 10 VARIABLES ARE SIGNIFICANT, so not dropping any of them##

##### Pre_processing ends here #######
 

###Fitting Random Forest on the data  #####

   
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred_rf=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print('Accuracy of  Random Forest classifier on test set: {:.2f}'.format(metrics.accuracy_score(y_test, y_pred_rf)))

print (pd.crosstab(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_rf))

### Feature's ranking for selecting important features in the data set ##

feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
print(feature_imp)

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

### From the graph, it CONFIRMS THAT ALL ATTRIBUTES ARE IMORTANT  #####

