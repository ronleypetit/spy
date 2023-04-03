import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.deterministic import DeterministicProcess

#plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 5),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/Users/Rohacky/Documents/ds projects/spy/spy.csv')
# print(df.head())

year = df[df.Date.str.contains('2022')]
# print(year.head())

# this is to create a time dummy to plot the closing price over time

year['Time'] = np.arange(len(year.index))

# this is to create a plot of the closing price from Jan to Dec 2022; we can see a clearly marked downward trend

fig, ax = plt.subplots()
ax.plot('Time', 'Close', data = year)
ax = sns.regplot(x = 'Time', y = 'Close', data = year, ci = None, scatter_kws = dict(color = '0.25'))
ax.set_title('Time plot of SPY Closing Price during 2022')
# plt.show()

# print(year.Close.loc[year.Date == '2022-01-03'])
# print(year.Close.loc[year.Date == '2022-12-30'])

"""
Let's now take a step back and look at the trend over the life of the index. From this perspective the decline over one 
year is just one part of an overall upward trend. Let's model this trend using Linear Regression and a third order polynomial
"""
# prepare X and y
y = df['Close']
dp = DeterministicProcess(index = y.index, order = 3)
X = dp.in_sample()
# fit the model
model = LinearRegression()
model.fit(X, y)
# make predictions
y_pred = pd.Series(model.predict(X), index = X.index)
# plot the predictions against the actual values
ax = y.plot(alpha = 0.5, title = "Time plot of SPY Closing Price over its history", ylabel = "Close")
ax = y_pred.plot(ax=ax, linewidth = 3, label = "Trend", color = 'C9')
ax.legend()
# plt.show()
print('Model score = ', model.score(X,y))
print('Mean Absolute Error = ', mean_absolute_error(y, y_pred))
print('Mean Absolute Percentage Error = ', mean_absolute_percentage_error(y, y_pred))

# actual current price compared to the price prediction of our model
print('Close date as of Feb 6: ', df.Close[df.Date == '2023-02-06'])
print('Model prediction: ', y_pred.tail(1))

# current price is some way below the long-term trend prediction
# one year forecast
future = dp.out_of_sample(steps = 260)
y_future = pd.Series(model.predict(future), index=future.index)
print('One year forecast: ', y_future.tail(1))
