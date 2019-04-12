# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:59:15 2019

@author: sayray
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset = pd.read_csv("NewPC.csv")
X= dataset.iloc[:,[0]].values
y= dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)


from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train);

y_pred = regressor.predict(X_test)

plt.plot(y_pred, c="Red")
plt.plot(y_test, c="Green")
plt.show()

#http://www.webgraphviz.com/
# import export_graphviz 
from sklearn.tree import export_graphviz  
  
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='treePC7.dot', 
               feature_names =['Average Salary'])
