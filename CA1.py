# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:55:10 2019

@author: sidhant
"""

import numpy as nm
import pandas as pd
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mse
from sklearn.metrics import r2_score,classification_report, confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt    
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

def GBRegressor(Xtrain,Xtest,Ytrain,Ytest):   
    model = GradientBoostingRegressor(n_estimators=100, max_depth=8,
                                        learning_rate=0.1, loss='huber',
                                        random_state=1)
    model.fit(Xtrain,Ytrain)
    y_test_predict=model.predict(Xtest)
    y_train_predict=model.predict(Xtrain)

    meansqr_test=mse(Ytest, y_test_predict)
    meansqr_train=mse(Ytrain, y_train_predict)
    rmse_test = (nm.sqrt(mean_squared_error(Ytest, y_test_predict)))
    rmse_train = (nm.sqrt(mean_squared_error(Ytrain, y_train_predict)))
    abc=r2_score(Ytest,y_test_predict)
    ab=model.score(Xtest,Ytest)
    print("###################GradientBoosting###############")
    print("accuracy=",ab)
    print("RMSE for test data=",rmse_test)
    print("RMSE for training data=",rmse_train)
    print("MSE for test data=",meansqr_test)
    print("MSE for training data=",meansqr_train)
    print("")
    print("")
    sns.regplot(Ytest, y_test_predict, fit_reg=True, scatter_kws={"s": 100})
    plt.show()
    #Snippet_165()
    return model,ab

def mlpregressor(Xtrain,Xtest,Ytrain,Ytest):    
    
    model = MLPRegressor(hidden_layer_sizes=(130,60,16,4),activation='relu',learning_rate='adaptive' )
    
    model.fit(Xtrain,Ytrain)
    #print(model)

        # make predictions
    expected_y  = Ytest
    y_test_predict=model.predict(Xtest)
    y_train_predict=model.predict(Xtrain)

    meansqr_test=mse(Ytest, y_test_predict)
    meansqr_train=mse(Ytrain, y_train_predict)
    rmse_test = (nm.sqrt(mean_squared_error(Ytest, y_test_predict)))
    rmse_train = (nm.sqrt(mean_squared_error(Ytrain, y_train_predict)))
    abc=r2_score(Ytest,y_test_predict)
    ab=model.score(Xtest,Ytest)
    print("###################NeuralNetwork###############")
    print("accuracy=",ab)
    print("RMSE for test data=",rmse_test)
    print("RMSE for training data=",rmse_train)
    print("MSE for test data=",meansqr_test)
    print("MSE for training data=",meansqr_train)
    print("")
    print("")
    sns.regplot(Ytest, y_test_predict, fit_reg=True, scatter_kws={"s": 100})
    plt.show()
    #Snippet_165()
    return model,ab



data= pd.read_csv("C:/sidhant/CA1/7210_1/7210_1.csv")
#print(data.describe())
data=data.drop(['id','asins','categories','count','dimension','descriptions','ean','flavors','imageURLs','isbn','keys','manufacturerNumber','prices.availability','prices.color','prices.count','prices.dateAdded','prices.dateSeen','prices.flavor','prices.offer','prices.returnPolicy','prices.source','prices.sourceURLs','prices.warranty','reviews','skus','sourceURLs','upc','vin','websiteIDs',],axis=1)

#null_columns=data.columns[data.isnull().any()]
#print((data['brand'].isnull()).head(20))
#print(data.info())

#############combine 2 column with their mean ##################################
price=data[['prices.amountMax','prices.amountMin']].mean( axis=1) #taking mean of 2 columns to get avprice
data=data.drop(['prices.amountMax','prices.amountMin'],axis=1)
data['averagePrice']=price
#############################END###############################################


#############brand column data preprocessing##################################
count=0
data.brand.fillna(data.manufacturer,inplace=True)
data=data.drop(['manufacturer'],axis=1)

#null_columns=data.columns[data.isnull().any()]
#print(data[null_columns].isnull().sum())
#print(data['brand'].value_counts().index[0])    #mode imputation 
encoder=ce.BinaryEncoder(cols=['brand'])
data=encoder.fit_transform(data)
#print(data['brand'])
#############################END##############################################


################colors column data preprocessing###############################
#print(data['colors'].describe())
#print((data['colors'].values =='').sum())#####check for null values in column

data=data.drop(data[data['colors']=='b,c,a,d'].index)
data=data.drop(data[data['colors']=='c,d,a,b'].index)
data=data.drop(data[data['colors']=='d,b,a,c'].index)
data=data.drop(data[data['colors']=='a,c,b'].index)
data.colors=data.colors.fillna(data['colors'].value_counts().index[0])   #mode imputation for blank values

#print((data.colors.values =='').sum())
#
#null_columns=data.columns[data.isnull().any()]
#print(data[null_columns].isnull().sum())#####check for null values in column


#print(data['colors'].unique())######check for unique value set
#print(len(data['colors'].unique()))######check for unique value set

encoder=ce.BinaryEncoder(cols=['colors'])
data=encoder.fit_transform(data)

######################END#####################################################

 
#######################merchants column data processing#######################
#print((data['prices.merchant']).count())
data= data.dropna(axis=0,subset=['prices.merchant'])
#print((data['prices.merchant']).count())
data['merchants']=data['prices.merchant']
data=data.drop(['prices.merchant'],axis=1)

encoder=ce.BinaryEncoder(cols=['merchants'])
data=encoder.fit_transform(data)
#data['merchants']= data['merchants'].astype('category')
#data['merchants']= data['merchants'].cat.codes

#######################END####################################################


#######################name column data processing############################

encoder=ce.BinaryEncoder(cols=['name'])
data=encoder.fit_transform(data)
#data['name']= data['name'].astype('category')
#data['name']= data['name'].cat.codes

#######################END####################################################


######################price.size & price.condition column data processing############################
data= data.dropna(axis=0,subset=['prices.condition'])
#data=data.dropna(subset=['prices.condition', 'prices.size'], how='all')
data['prices.size']=data['prices.size'].fillna(data['prices.size'].value_counts().index[0])
#print(data['prices.condition'].unique())


encoder=ce.BinaryEncoder(cols=['prices.size'])
data=encoder.fit_transform(data)

#data['prices.size']= data['prices.size'].astype('category')
#data['prices.size']= data['prices.size'].cat.codes
encoder=ce.BinaryEncoder(cols=['prices.condition'])
data=encoder.fit_transform(data)
#data['prices.condition']= data['prices.condition'].astype('category')
#data['prices.condition']= data['prices.condition'].cat.codes



#######################END####################################################


###############################prices.isSale###################################
encoder=ce.BinaryEncoder(cols=['prices.currency'])
data=encoder.fit_transform(data)

data['prices.isSale']= data['prices.isSale'].astype('category')
data['prices.isSale']= data['prices.isSale'].cat.codes


data=data.drop(['dateAdded','dateUpdated','features','weight','sizes','prices.shipping','quantities'],axis=1)

data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

#################################END###########################################

X=data.drop(['averagePrice'],axis=1)
Y=data['averagePrice']

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.25,random_state=3)
#null_columns=data_copy.columns[data_copy.isnull().any()]
#print((data_copy[null_columns].isnull().sum()))
print("###################LinearRegression####################")
#model=LinearRegression()
#model.fit(Xtrain,Ytrain)
#
#y_test_pridict=model.predict(Xtest)
#y_train_pridict=model.predict(Xtrain)
#meansqr_test=mse(Ytest, y_test_pridict)
#meansqr_train=mse(Ytrain, y_train_pridict)
#rmse_test = (nm.sqrt(mean_squared_error(Ytest, y_test_pridict)))
#rmse_train = (nm.sqrt(mean_squared_error(Ytrain, y_train_pridict)))
#abc=r2_score(Ytest,y_test_pridict)
#ab=model.score(Xtest,Ytest)
#
#
#
#print("accuracy=",ab)
#print("RMSE for test data=",rmse_test)
#print("RMSE for training data=",rmse_train)
#print("MSE for test data=",meansqr_test)
#print("MSE for training data=",meansqr_train)
#print("")
#print("")
#
#plt.scatter(Xtest, Ytest,  color='black')
#plt.plot(Xtest, y_train_pridict, color='blue', linewidth=3)
#
#plt.xticks(())
#plt.yticks(())
#
#plt.show()
#print("###################RegressionTree###############")
#      
#      
#model_tree= DecisionTreeRegressor(max_depth=9)
#model_tree.fit(Xtrain,Ytrain)
#pridiction_tree=model_tree.predict(Xtest)
#pridiction_tree_train=model_tree.predict(Xtrain)
#rmse_tree_test = (nm.sqrt(mean_squared_error(Ytest, pridiction_tree)))
#rmse_tree_train = (nm.sqrt(mean_squared_error(Ytrain, pridiction_tree_train)))
#meansqr_tree=mse(Ytest, pridiction_tree)
#meansqr_tree_train=mse(Ytrain, pridiction_tree_train)
#
#acc=r2_score(Ytest, pridiction_tree)
#print("RMSE for test data=",rmse_tree_test)
#print("RMSE for training data=",rmse_tree_train)
#print("MSE for regression tree model =",meansqr_tree)
#print("MSE for regression tree model train =",meansqr_tree_train)
#print("accuracy=",acc)

#model=mlpregressor(Xtrain,Xtest,Ytrain,Ytest)
model=GBRegressor(Xtrain,Xtest,Ytrain,Ytest)



#print("###############NeuralNet###############################")
#
#X_NN=data.drop(['prices.isSale'],axis=1)  
#Y_NN=data['prices.isSale']
#
#Xtrain_NN,Xtest_NN,Ytrain_NN,Ytest_NN=train_test_split(X_NN,Y_NN,test_size=0.25,random_state=3)
#
#mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000,verbose=2)  
#mlp.fit(Xtrain_NN, Ytrain_NN)  
#
#predictions = mlp.predict(Xtest_NN)  
#
#
#print("Accuracy", metrics.accuracy_score(Ytest_NN, predictions))
#print(confusion_matrix(Ytest_NN,predictions))  
#print(classification_report(Ytest_NN,predictions))
#
#rmse_NNtest = (nm.sqrt(mean_squared_error(Ytest_NN, predictions)))
#meansqr_NNtest=mse(Ytest_NN, predictions)
#print("RMSE =",rmse_NNtest)
#print("MSE =",meansqr_NNtest)
#
#print("Accuracy on training set: {:.3f}".format(mlp.score(Xtrain_NN, Ytrain_NN)))
#print("Accuracy on test set: {:.3f}".format(mlp.score(Xtest_NN, Ytest_NN)))





