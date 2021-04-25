#Partners: Alexandra Fernandez & Amanda Condron
#Final Project: Part 2

#Import statements
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

#Loading csv into a pandas dataframe
import pandas as pd
df = pd.read_csv("billionaires.csv", sep=",")

#For testing: to display all the columns in terminal
pd.set_option("display.max_rows", None, "display.max_columns", None)

#Printing first three rows to verify dataframe looks correct
print("First three rows:")
print(df.head(3))

#Utilizing label encoder to re-encode all our strings as integers
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

#Rounding to nearest billion for wealth.worth in billions
df['wealth.worth in billions'] = df['wealth.worth in billions'].apply(lambda x: round(x, 0))
df['wealth.worth in billions'] = df['wealth.worth in billions'].astype(int)

#Rounding to the nearest integer for location gdp
df['location.gdp'] = df['location.gdp'].apply(lambda x: round(x, 0))
df['v'] = df['location.gdp'].astype(int)

#Turning the Dataframe into a numpy array
numpy_array = df.to_numpy()
print("numpy array shape = ", numpy_array.shape)


#For testing to see all the values in numpy array in terminal
np.set_printoptions(threshold=sys.maxsize)

#Splitting our data into input features X and output Y
y = numpy_array[:,19]
x_df = df.drop('wealth.how.inherited', axis=1)
x = x_df.to_numpy()

# Random state = 'none' so that we get the same results
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

#First Technique:
#Multinomial Naive Bayes Classifier
nb_classifier = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
nb_classifier.fit(X_train, y_train)
print("The mean accuracy on the given training data and labels", nb_classifier.score(X_train,y_train))
print("The mean accuracy on the given test data and labels", nb_classifier.score(X_test,y_test))

#Part 3 for the First Technique
#parameter_grid dictionary
nb_param_grid_all_featurues = {'alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001), 'fit_prior': (True, False), 'class_prior':([None])}

#GridSearchCV object for the First Technique
nb_search_all_features = GridSearchCV(nb_classifier, nb_param_grid_all_featurues, cv= 5)
nb_search_all_features.fit(X_train, y_train)
print("Best Score for First Technique:", nb_search_all_features.best_score_)
print("Best Params for First Technique:", nb_search_all_features.best_params_)

#Using the obtained hyperparameter values to fit and score our basic model on our testing data.
nb_classifier_optimal = MultinomialNB(alpha=0.01, fit_prior=True, class_prior=None)
nb_classifier_optimal.fit(X_train, y_train)
print("The mean accuracy on the given test data and labels (using optimal parameters)", nb_classifier_optimal.score(X_test,y_test))

#Second Technique:
#First Using Extra Trees Classifier to see the importance of the features
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X_train, y_train)
#print("The importance of each feature: ", model.feature_importances_)

#Dropping all columns with a value of ~0.02 or less
df = df.drop(columns=["year", "company.type", "location.gdp", "wealth.how.was political","company.relationship","location.citizenship","location.country code","location.region","wealth.type"])
new_numpy_array = df.to_numpy()

#Splitting our data into new input features x2
x_df2 = df.drop('wealth.how.inherited', axis=1)
x2 = x_df2.to_numpy()

#Splitting the data into training and testing datasets
X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y, test_size=0.25, random_state=42)
nb_classifier = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
nb_classifier.fit(X_train2, y_train2)
print("The mean accuracy on the given training data and labels after feature selection", nb_classifier.score(X_train2,y_train2))
print("The mean accuracy on the give test data and labels after feature selection", nb_classifier.score(X_test2,y_test2))

#Part 3:
#parameter_grid dictionary
nb_param_grid_select_featurues = {'alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001), 'fit_prior': (True, False), 'class_prior':([None])}

#GridSearchCV object for the basic method
nb_search_select_features = GridSearchCV(nb_classifier, nb_param_grid_all_featurues, cv= 5)
nb_search_select_features.fit(X_train2, y_train2)
print("Best Score for Second Technique:", nb_search_select_features.best_score_)
print("Best Params for Second Technique:", nb_search_select_features.best_params_)

#Using the obtained hyperparameter values to fit and score our advanced model on our testing data.
nb_classifier_optimal2 = MultinomialNB(alpha=0.01, fit_prior=True, class_prior=None)
nb_classifier_optimal2.fit(X_train2, y_train2)
print("The mean accuracy on the given test data and labels (using optimal parameters)", nb_classifier_optimal2.score(X_test2,y_test2))
