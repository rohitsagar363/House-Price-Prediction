# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:43:03 2020

@author: rohith
"""


#importing libraries
import pandas as pd
from sklearn.metrics import mean_absolute_error


#importing dataset
home_data = pd.read_csv('train.csv')


#extracting features and labels
y = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]


# Split into validation and training data
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#Training the model on training set
from sklearn.tree import DecisionTreeRegressor
# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))
#accuracy
print(iowa_model.score(val_X,val_y))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))
#accuracy
print(iowa_model.score(val_X,val_y))


from sklearn.ensemble import RandomForestRegressor
# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state = 1)
# fit your model
rf_model.fit(train_X,train_y)
rf_pred = rf_model.predict(val_X)
# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(val_y,rf_pred)
print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
#accuracy
print(rf_model.score(val_X,val_y))

'''
Achieved an accuracy of 84% using Random Forest Model
'''

from sklearn.ensemble import GradientBoostingRegressor
# Define the model. Set random_state to 1
gb_model = GradientBoostingRegressor(random_state=1)
# fit your model
gb_model.fit(train_X,train_y)
gb_pred = gb_model.predict(val_X)
# Calculate the mean absolute error of your Random Forest model on the validation data
gb_val_mae = mean_absolute_error(val_y,rf_pred)
print("Validation MAE for GradientBoosting Model: {}".format(gb_val_mae))
#accuracy
print(gb_model.score(val_X,val_y))

'''
Achieved an accuracy of 85% using GradientBoosting Model
'''



