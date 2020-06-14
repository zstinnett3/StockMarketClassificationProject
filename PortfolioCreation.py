#!/usr/bin/env python
# coding: utf-8

# In[1]:


# method name: Portfolio Creation
# purpose: This script uses the data from the predictions to make a portfolio. 
# created: 6/14/2020 11:01PM
# Author: Zack Stinnett
# Revisions: Added logging       6/14/2020 7:30PM


# In[2]:


import pandas as pd
import random
import logging


# In[3]:


logging.basicConfig(filename='portfoliocreation.log', level=logging.INFO,
                    format='%(levelname)s:%(message)s')


# In[4]:


df = pd.read_csv('StockDataWithPredictions.csv')


# In[5]:


symbols_preds_opens = df[['Symbol', 'Prediction', 'Open', 'Signal']]


# In[6]:


# Getting the first time a stock appeared
first_occurances = symbols_preds_opens.drop_duplicates(['Symbol'], keep='first')


# In[7]:


first_occurances_with_buys = first_occurances[(first_occurances['Prediction'] == 1.0)]


# In[8]:


print('The number of stocks that had a buy on first trigger: {}.'.format(len(first_occurances_with_buys)))


# In[9]:


long_stocks = random.choices(list(first_occurances_with_buys['Symbol']), k=10)


# In[10]:


first_occurances[first_occurances['Symbol'].isin(long_stocks)]


# In[11]:


# Considering the time period the data is taking place, March to June 2020. I chose a long portfolio, 
# since that was a great buying opportunity. 


# In[12]:


# The following is a great article on portfolio creation with Python: https://medium.com/@randerson112358/python-for-finance-portfolio-optimization-66882498847
# The process would likely need more than a couple months of data to be effective in this situation though. 


# In[13]:


logging.info('Script has finished')

