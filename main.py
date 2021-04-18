#part 2 number 2:


import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

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
df['demographics.age'] = df['demographics.age'].abs()
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

# rounding to nearest billion for wealth.worth in billions
df['wealth.worth in billions'] = df['wealth.worth in billions'].apply(lambda x: round(x, 0))
df['wealth.worth in billions'] = df['wealth.worth in billions'].astype(int)

df['location.gdp'] = df['location.gdp'].apply(lambda x: round(x, 0))
df['v'] = df['location.gdp'].astype(int)

print(df.head(15))
# print(df.dtypes)


numpy_array = df.to_numpy()
# print(numpy_array)
print(numpy_array.shape)
np.set_printoptions(threshold=sys.maxsize)
print(numpy_array>=0)

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

print("printing shapes")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#part 5: apply basic technique
#how many categories the other features have and use the average of that. -- for bins for #wealth.worth in billions
# col1Count = df['name'].nunique()
# col2Count = df['rank'].nunique()
# col3Count = df['year'].nunique()
# col4Count = df['company.founded'].nunique()
# col5Count = df['company.name'].nunique()
# col6Count = df['company.relationship'].nunique()
# col7Count = df['company.sector'].nunique()
# col8Count = df['company.type'].nunique()
# col9Count = df['demographics.age'].nunique()
# col10Count = df['demographics.gender'].nunique()
# col11Count = df['location.citizenship'].nunique()
# col12Count = df['location.country code'].nunique()
# col13Count = df['location.region'].nunique()
# col14Count = df['wealth.type'].nunique()
# col15Count = df['wealth.how.category'].nunique()
# col16Count = df['wealth.how.from emerging'].nunique()
# col17Count = df['wealth.how.industry'].nunique()
# col18Count = df['wealth.how.inherited'].nunique()
# col19Count = df['wealth.how.was founder'].nunique()
# col20Count = df['wealth.how.was political'].nunique()
# col20Count = df['location.gdp'].nunique()
#
# print(col1Count)
# print(col2Count)




# nb_classifier = MultinomialNB()
# y_prediction = nb_classifier.fit(X_train, y_train).predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_prediction).sum()))

#part 5:
nb_classifier = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
nb_classifier.fit(X_train, y_train)
print("mean accuracy on given train data and labels", nb_classifier.score(X_train,y_train))
print("mean accuracy on give test data and labels", nb_classifier.score(X_test,y_test))

#