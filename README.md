# StockMarketClassificationProject
Using a few months worth of daily data, I have classified good trade entry points and ultimately 10 stocks to buy

First thing, when downloading this repo, save the Indicators.py to a folder in your working directory called Functions to allow the import to work. Or change the import Functions.Indicators as ind to reflect where you store the file. 

If you want to use the stacking algorithms, you will need to run the StockMarketClassification file in an environment with the latest versions. But the StockMarketDataPreprocessing needs to use the older version due to how some of the Indicators are written. 

    Order to run files: 
StockMarketDataPreprocessing, then StockMarketClassification, then PortfolioCreation. 

    StockMarketDataPreprocessing
In this file, I took data from a MongoDB instance and first found 100 stocks that didn't have too many nulls and also didn't have negative prices. Some stocks had negative prices surprisingly. Once I had a set of stocks I filled in the nulls with cubic splines, because these splines had the necessary flexibility to accurately fill in the null values with reasonable approximate answers. After doing this, I added a signal column showing whether one should buy, short, or do nothing at each time step. The signal was determined by whether the stock went up or down 5%. This amount roughly balanced the classes, but also in a real world since gives leeway for commissions and slippage. I next added indicators. Be aware of the version of your environment, because the newest versions of pandas don't support ix anymore and the Indicators.py file has some of that old syntax. Once the indicators are added, the classes are more fully balanced. The results are then saved to a csv for later processing. 
    
    StockMarketClassification
First the stock symbols are encoded and the data scaled. Next it is split into training and test sets. I did not use the traditional splitting functions that python has, because these lead to leakage. I split the training and test on date to try to reduce leakage as best as possible. Next the classification alorigthms are trained through a cross validation process. I originally wanted use a stacking ensemble to get the best of several models, but from results it appears this isn't a good model. The random forest model ended up being the best with nearly 90% accuracy during training. Once the classification is done and the model is retested on some unseen test data, I re-added the labels and unscaled the test data. Next the data is saved for use in the next step.
    
    PortfolioCreation
This part uses the past results to create a long portfolio recommendation. Considering the time span, this was the best recommendation.
    
    Possible Improvements
The algorthms would likely benefit from more data and cleaner data. Considering the chaos in the markets during the timespan, it is very likely the cubic splines didn't match the actual results as well as could have during more stable times. More indicators might be helpful. I have read some studies using as many as 70. With more data, neural networks might outperform the random forests. Also, testing on unscaled data might be worth exploring. 
