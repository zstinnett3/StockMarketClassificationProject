#!/usr/bin/env python
# coding: utf-8

# In[1]:


# method name: Stock Market Classification Algorithm
# purpose: This script uses data from the previous data preprocessing algorithm to classify possible trades
# created: 6/14/2020 5:01PM
# Author: Zack Stinnett
# Revisions: Added logging       6/14/2020 7:30PM


# In[2]:


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import logging


# In[3]:


logging.basicConfig(filename='stockmarketclassificationalgo.log', level=logging.INFO,
                    format='%(levelname)s:%(message)s')


# In[4]:


stock_data = pd.read_csv('DataPreprocessingResults.csv')


# In[5]:


#Encoding stock symbols with numerical values
label_encoder = LabelEncoder()


# In[6]:


stock_data['Symbol'] = label_encoder.fit_transform(stock_data['Symbol'])

logging.info('Stock Data has been pulled in and labeled encoded')


# In[7]:


# Need to scale values in the data to allow the classification algorithms to work better. 
from sklearn.preprocessing import StandardScaler
def rescale(data):
    data = data.dropna().astype('float')
    scaler = StandardScaler()
    return scaler.fit_transform(data)


# In[8]:


# Fixing infinite values before using scaler
# https://stackoverflow.com/questions/31323499/sklearn-error-valueerror-input-contains-nan-infinity-or-a-value-too-large-for

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


# In[9]:


# Splitting manually on a date vs randomly to avoid leakage
train_data = stock_data[stock_data['Date']<'2020-06-03']
test_data = stock_data[stock_data['Date']>='2020-06-03']

train_data.drop(['Date'], axis=1, inplace=True)
test_data.drop(['Date'], axis=1, inplace=True)

train_data = clean_dataset(train_data)
test_data = clean_dataset(test_data)

train_X, train_y = train_data.drop(['Signals'],axis=1), train_data['Signals']
test_X, test_y   = test_data.drop(['Signals'],axis=1),  test_data['Signals']


# In[10]:


scaled_train_X = rescale(train_X)
scaled_test_X = rescale(test_X)


# In[11]:


# Version needs to be at least 0.22 for the ensemble stacking to work.
import sklearn
print(sklearn.__version__)


# In[12]:


# Code for the following comes from the stacking section from: https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
# Was introduced to the idea in Advances in Financial Machine Learning by Marcos Lopez de Prado 


# In[13]:


def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('lr', LogisticRegression()))     #Analysis showed this model underperformed
    level0.append(('knn', KNeighborsClassifier()))   #Analysis showed this model underperformed
    level0.append(('rf_1',RandomForestClassifier(class_weight='balanced')))
    level0.append(('rf_2',RandomForestClassifier(class_weight='balanced_subsample')))
    level0.append(('cart', DecisionTreeClassifier()))
    
    # define meta learner model
    level1 = RandomForestClassifier()
    
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model


# In[14]:


# get a list of models to evaluate
def get_models():
    models = dict()
    #models['lr'] = LogisticRegression()     #Analysis showed this model underperformed
    #models['knn'] = KNeighborsClassifier()   #Analysis showed this model underperformed
    models['rf_1'] = RandomForestClassifier(class_weight='balanced')
    models['rf_2'] = RandomForestClassifier(class_weight='balanced_subsample')
    models['xgb'] = XGBClassifier(num_classes=3)
    #models['bayes'] =  GaussianNB()        #Analysis showed this model underperformed
    models['stacking'] = get_stacking()
    return models


# In[15]:


# evaluate a give model using cross-validation
def evaluate_model(model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, scaled_train_X, train_y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# In[16]:


models = get_models()


# In[17]:


results, names = list(), list()

for name, model in models.items():
    scores = evaluate_model(model)
    results.append(scores)
    names.append(name)
    print('Model {} had {:.0%} training accuracy with a {:.0} standard deviation.'.format(name, np.mean(scores), np.std(scores)))


# In[18]:


# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


# In[19]:


# The stacking algorithm ended up not performing nearly as well as the random forest algorithms. 


# In[20]:


# random forest model creation
score = evaluate_model(models['rf_1'])
print('Model {} had {:.0%} training accuracy with a {:.0} standard deviation.'.format(name, np.mean(score), np.std(score)))


# In[21]:


print("=== All AUC Scores ===")
print(score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", score.mean())


# In[22]:


# Didn't do any hyperparameter turning since the AUC is already close to 0.90, indicating a great model already
# Now to create a portfolio from the predictions after saving this data.


# In[23]:


y_pred = cross_val_predict(model, scaled_test_X, test_y, cv=10)


# In[24]:


print('The Cross Validated Prediction Accuracy was: {:.0%}'.format(accuracy_score(test_y, y_pred)))


# In[25]:


test_X['Prediction'] = y_pred


# In[26]:


test_X['Symbol'] = label_encoder.inverse_transform(test_X['Symbol'].astype(int))


# In[27]:


test_X['Signal'] = test_y


# In[28]:


test_X.to_csv('StockDataWithPredictions.csv')


# In[29]:


logging.info('CLassification algorithms have finished and data is ready for portfolio.')

