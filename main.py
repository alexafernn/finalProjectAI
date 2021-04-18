#part 2 number 2:

#import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys

#load into a numpy array and print shape
import pandas as pd
df = pd.read_csv("billionaires.csv", sep=",")

#Part4:  your input or output features are categorical and represented as strings,
# some models my require you to re-encode them as integers. In this case, add the line
# from sklearn.preprocessing import LabelEncoder to the top of your Python file to import the label encoder.
# Refer to the documentation and fit and apply a label encoder to each column of the data as needed.
# You should apply this transformation before your test/train split in Item 3.

#columns we need to change from string
pd.set_option("display.max_rows", None, "display.max_columns", None)
le = LabelEncoder()
df['name'] = le.fit_transform(df['name'])
df['company.name'] = le.fit_transform(df['company.name'].astype(str))
df['company.relationship'] = le.fit_transform(df['company.relationship'].astype(str))
df['company.sector'] = le.fit_transform(df['company.sector'].astype(str))
df['company.type'] = le.fit_transform(df['company.type'].astype(str))
df['demographics.gender'] = le.fit_transform(df['demographics.gender'].astype(str))
df['location.citizenship'] = le.fit_transform(df['location.citizenship'].astype(str))
df['location.country code'] = le.fit_transform(df['location.country code'].astype(str))
df['location.region'] = le.fit_transform(df['location.region'].astype(str))
df['wealth.type'] = le.fit_transform(df['wealth.type'].astype(str))
df['wealth.how.category'] = le.fit_transform(df['wealth.how.category'].astype(str))
df['wealth.how.from emerging'] = le.fit_transform(df['wealth.how.from emerging'].astype(str))
df['wealth.how.industry'] = le.fit_transform(df['wealth.how.industry'].astype(str))
df['wealth.how.inherited'] = le.fit_transform(df['wealth.how.inherited'].astype(str))
df['wealth.how.was founder'] = le.fit_transform(df['wealth.how.was founder'].astype(str))
df['wealth.how.was political'] = le.fit_transform(df['wealth.how.was political'].astype(str))

print(df.head(5))


numpy_array = df.to_numpy()
# print(numpy_array)
print(numpy_array.shape)

#if you are applying supervised learning, split your data into input features X and output Y
print("about to print y aka outputs")
y = numpy_array[:,19]
# print(y)
print(y.shape)

print("about to print x aka input")
x_df = df.drop('wealth.how.inherited', axis=1)
x = x_df.to_numpy()
print(x.shape)

#Part 3: Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)



