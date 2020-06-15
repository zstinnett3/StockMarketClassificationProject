#!/usr/bin/env python
# coding: utf-8


# method name: Portfolio Creation
# purpose: This script uses the data from the predictions to make a portfolio. 
# created: 6/14/2020 11:01PM
# Author: Zack Stinnett
# Revisions: Added logging       6/14/2020 7:30PM

import pandas as pd
import random
import logging


logging.basicConfig(filename='portfoliocreation.log', level=logging.INFO,
                    format='%(levelname)s:%(message)s')


df = pd.read_csv('StockDataWithPredictions.csv')


symbols_preds_opens = df[['Symbol', 'Prediction', 'Open', 'Signal']]

# Getting the first time a stock appeared
first_occurances = symbols_preds_opens.drop_duplicates(['Symbol'], keep='first')


first_occurances_with_buys = first_occurances[(first_occurances['Prediction'] == 1.0)]


print('The number of stocks that had a buy on first trigger: {}.'.format(len(first_occurances_with_buys)))


long_stocks = random.choices(list(first_occurances_with_buys['Symbol']), k=10)


first_occurances[first_occurances['Symbol'].isin(long_stocks)]


# Considering the time period the data is taking place, March to June 2020. I chose a long portfolio, 
# since that was a great buying opportunity. 


# The following is a great article on portfolio creation with Python: https://medium.com/@randerson112358/python-for-finance-portfolio-optimization-66882498847
# The process would likely need more than a couple months of data to be effective in this situation though. 


logging.info('Script has finished')

