# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:23:46 2019

@author: sidhant
"""

import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_excel("C:\sidhant\CA1\summary.xlsx")
data=data.dropna(axis=0)
print(data)
#index=len(data['Model'])
#plt.bar(data['Model'],data['Accuracy %'])
#plt.xlabel(data['Model'], fontsize=5)
#plt.ylabel(data['Accuracy %'])
#plt.show()
data.plot.bar(x='Model',y=['Accuracy %','RMSE','MSE'], title='Model performance')
#data.plot.bar(x='Model',y='Accuracy %')