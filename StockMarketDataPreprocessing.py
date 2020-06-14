#!/usr/bin/env python
# coding: utf-8

# In[1]:


# method name: Stock Market Data Preprocessing
# purpose: This script pulls down daily data from a MongoDB instance, then processes to data for use in ML algos
# created: 6/14/2020 12:01PM
# Author: Zack Stinnett
# Revisions: Added logging       6/14/2020 7:30PM


# In[2]:


# A big issue that was encountered is the indicators were developed using an older version that allowed ix and other 
# functions that are not longer supported, but the newest version of sklearn is required downstream
# To run this, use an environment with older versions.
# Analysis showed the stacking algorithm may not be useful, so changing environments is unnecessary.
# Be sure to comment out the stacking algorithm lines in the next script if using an older environment. 


# In[3]:


import numpy as np
import pandas as pd
import Functions.RetrieveIntradayData as ret
import Functions.Indicators as ind
from inspect import getmembers, isfunction
import pymongo
import concurrent.futures
import datetime
from sklearn.utils import resample
import logging


# In[4]:


logging.basicConfig(filename='stockmarketdatapreprocessing.log', level=logging.INFO,
                    format='%(levelname)s:%(message)s')


# In[5]:


mongo_client = pymongo.MongoClient('mongodb://mlcandidates:crackthecode@100.2.158.147:27017/')
finDb = mongo_client['findata']


# In[6]:


dailyCollection = finDb['day']


# In[7]:


# To get data for all symbols
all_stocks_daily_df = pd.DataFrame(list(dailyCollection.find({'close':{'$ne':'NaN'}})))


# In[8]:


# Number of stock symbols
print('The number of available stocks is {}.'.format(len(all_stocks_daily_df.Symbol.unique())))


# In[9]:


# Next let's find out the number of null close prices
# and try to choose stocks with lower null counts and non negative prices
stock_null_percentages = {}
stock_time_lengths = {}

def get_close_null_percentage(symbol):
    daily_df = pd.DataFrame(list(dailyCollection.find({'Symbol':symbol, 'close':{'$ne': 'NaN'}})))
    stock_null_percentages[symbol] = daily_df['Close'].isnull().sum() / len(daily_df)
    stock_time_lengths[symbol] = len(daily_df), len(daily_df[daily_df['Close']>0])


# In[10]:


# Using multithreading to speed up data gathering
symbols = all_stocks_daily_df.Symbol.unique()

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(get_close_null_percentage, symbols)

logging.info('Stock Data has been pulled in')


# In[11]:


stocks_with_max_lengths = {k: v for k, v in stock_time_lengths.items() if (v[0]>75) and (v[1]>0)}
stock_symbols_with_max_length_list = list(stocks_with_max_lengths.keys())
null_percentage_with_max_len = {key: value for key, value in stock_null_percentages.items() if key in stock_symbols_with_max_length_list}


# In[12]:


# There is a lot of nulls in this data, but most stocks only are missing 32% of the closes. 
# So let's get the top 100 stocks to analyze
import operator
top_stocks = sorted(null_percentage_with_max_len.items(), key=operator.itemgetter(1))[:100]
stocks_to_analyze = [i[0] for i in top_stocks]


# In[13]:


# I am using cubic splines to fill in the nulls in due to their flexibility in fit complex data

def create_interpolated_stock_df(symbol):
    daily_df = pd.DataFrame(list(dailyCollection.find({'Symbol':symbol, 'close':{'$ne': 'NaN'}})))
    daily_df.drop(['_id','Dividends','Stock Splits'], axis=1, inplace=True)   #At this scale, this data shouldn't matter
    daily_df['volume_change'].interpolate(method='spline', order=3, inplace=True, limit_direction="both")
    daily_df['volume_score'].interpolate(method='spline', order=3, inplace=True, limit_direction="both")
    daily_df['bullish'].interpolate(method='spline', order=3, inplace=True, limit_direction="both")
    daily_df['bearish'].interpolate(method='spline', order=3, inplace=True, limit_direction="both")
    daily_df['Open'].interpolate(method='spline', order=3, inplace=True, limit_direction="both")
    daily_df['High'].interpolate(method='spline', order=3, inplace=True, limit_direction="both")
    daily_df['Low'].interpolate(method='spline', order=3, inplace=True, limit_direction="both")
    daily_df['Close'].interpolate(method='spline', order=3, inplace=True, limit_direction="both")
    daily_df['Volume'].interpolate(method='spline', order=3, inplace=True, limit_direction="both")
    daily_df['DayofWeek'] = daily_df['Date'].apply(lambda x: x.weekday())
    return daily_df


# In[14]:


stock_dfs = [create_interpolated_stock_df(i) for i in stocks_to_analyze]


# In[15]:


# Predicting ahead by 5 days
# Choose 5% change bc it made the buys, sell, do nothing categories more balanced
horizon_threshold = 5
price_threshold = 0.05


# In[16]:


# Determine if should buy or sell or do nothing on the asset at each Open.
def BuyorSell(df):
    signals = []
    df = df.sort_values(by=['Date'])
    df.reset_index(drop=True, inplace=True)
    for index,open_value in enumerate(df['Open']):
        high = max(list(df['High'].iloc[index : index+horizon_threshold]))
        low =  min(list(df['Low'].iloc[index : index+horizon_threshold]))
        
        high_pct_diff = (high - open_value)/open_value
        low_pct_diff = (open_value-low)/open_value
        
        if (high_pct_diff>low_pct_diff) and (high_pct_diff>=price_threshold):
            signals.append(1)
        elif (high_pct_diff<low_pct_diff) and (low_pct_diff>=price_threshold):
            signals.append(-1)
        else:
            signals.append(0)
            
    df['Signals'] = np.array(signals)
    return df


# In[17]:


stocks_with_signals_list = [BuyorSell(i) for i in stock_dfs]


# In[18]:


# I created a .py with all the indicator functions on. Here I am pulling it in. 

indicators = [o for o in getmembers(ind) if isfunction(o[1])]
data = []
for dataframe in stocks_with_signals_list:
    for name, indicator in indicators:
        dataframe = indicator(dataframe)
    data.append(dataframe)


# In[19]:


StocksWithSignalsAndIndicators = pd.concat(data, ignore_index=True, sort=False)


# In[20]:


StocksWithSignalsAndIndicators['Signals'].value_counts()


# In[21]:


# Even though the classes are pretty balanced, let's upsample to make them more balanced.
# fixing imbalances classes from: https://elitedatascience.com/imbalanced-classes


# In[22]:


# Separate majority and minority classes and a middle
df_majority = StocksWithSignalsAndIndicators[StocksWithSignalsAndIndicators.Signals==1]
df_minority = StocksWithSignalsAndIndicators[StocksWithSignalsAndIndicators.Signals==0]
df_middle   = StocksWithSignalsAndIndicators[StocksWithSignalsAndIndicators.Signals==-1]


# In[23]:


# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=123) # reproducible results


# In[24]:


# Upsample middle class
df_middle_upsampled = resample(df_middle, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=123) # reproducible results


# In[25]:


# Combine majority class with upsampled minority class
StocksWithSignalsAndIndicators_upsampled = pd.concat([df_majority, df_minority_upsampled, df_middle_upsampled])


# In[26]:


# Display new class counts
StocksWithSignalsAndIndicators_upsampled.Signals.value_counts()


# In[27]:


StocksWithSignalsAndIndicators_upsampled.fillna(method='ffill', inplace=True)


# In[28]:


# Save the data so far for later use in the classification algorithms. 
StocksWithSignalsAndIndicators_upsampled.to_csv('DataPreprocessingResults.csv')

logging.info('Script has finished and csv has been created')

