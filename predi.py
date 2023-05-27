from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import random
from pymongo import MongoClient
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder




df = pd.read_csv('data.csv')
df.head()

df.describe()

# datatype info
df.info()

# find unique values
df.apply(lambda x: len(x.unique()))

# distplot for purchase
plt.style.use('fivethirtyeight')
plt.figure(figsize=(13, 7))
sns.distplot(df['Quantity'], bins=25)

# distribution of numeric variables
sns.countplot(df['Product_name'])

sns.countplot(df['Quantity'])

sns.countplot(df['Sales_per_week'])

sns.countplot(df['Sales_person'])

sns.countplot(df['Purchase_date'])

# bivariate analysis
occupation_plot = df.pivot_table(index='Sales_person', values='Quantity', aggfunc=np.mean)
occupation_plot.plot(kind='bar', figsize=(13, 7))
plt.xlabel('Sales_person')
plt.ylabel("Quantity")
plt.title("Sales_person and Quantity Analysis")
plt.xticks(rotation=0)
plt.show()

# check for null values
df.isnull().sum()

# to improve the metric use one hot encoding
# label encoding
cols = ['Product_category', 'City', 'Sales_person']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])
df.head()

corr = df.corr()
plt.figure(figsize=(14, 7))
sns.heatmap(corr, annot=True, cmap='coolwarm')
df.head()

X = df.drop(columns=['Product_id','Product_name', 'Product_category', 'Purchase_date','City_code'])
y = df['Quantity']

def train(model, X, y):
    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
    model.fit(x_train, y_train)

    # predict the results
    pred = model.predict(x_test)

    # cross validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))

    print("Results")
    print("MSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("CV Score:", np.sqrt(cv_score))

model = RandomForestRegressor(n_jobs=-1)
train(model, X, y)

features = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
features.plot(kind='bar', title='Feature Importance')

# Define x_test before making predictions
x_test = df.drop(columns=['Product_id','Product_name', 'Product_category', 'Purchase_date','City_code'])
pred = model.predict(x_test)

submission = pd.DataFrame()
submission['Product_id'] = df['Product_id']
submission['Sales_per_week'] = pred
submission.to_csv('submission.csv', index=False)
