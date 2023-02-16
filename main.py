import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

"""Dataframe column info"""
# 1. CRIM     per capita crime rate by town
# 2. ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
# 3. INDUS    proportion of non-retail business acres per town
# 4. CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 5. NOX      nitric oxides concentration (parts per 10 million)
# 6. RM       average number of rooms per dwelling
# 7. AGE      proportion of owner-occupied units built prior to 1940
# 8. DIS      weighted distances to five Boston employment centres
# 9. RAD      index of accessibility to radial highways
# 10. TAX      full-value property-tax rate per $10,000
# 11. PTRATIO  pupil-teacher ratio by town
# 12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 13. LSTAT    % lower status of the population
# 14. PRICE     Median value of owner-occupied homes in $1000's

"""The first column in the .csv file just has the row numbers, so it will be used as the index."""
pd.options.display.float_format = "{:,.2f}".format
data = pd.read_csv('boston.csv', index_col=0)
# print(data.shape)
# print(data.head())

"""Check for Missing Values and Duplicates"""
# data.info()

nan_values = data.isna().values.any()
# print(f'Any NaN values? {nan_values}')

duplicate_values = data.duplicated().values.any()
# print(f'Any duplicates? {duplicate_values}')

"""How many students are there per teacher on average? What is the average price of a home in the dataset?"""
# print(data.describe())

"""Use Seaborn's .displot() to create a bar chart and superimpose the Kernel Density Estimate (KDE) for the following variables: PRICE, RM, DIS, RAD"""
# House Prices
# sns.displot(data['PRICE'],
#             bins=50,
#             aspect=2,
#             kde=True,
#             color='#2196f3')
#
# plt.title(f'1970s Home Values in Boston. Average: ${(1000*data.PRICE.mean()):.6}')
# plt.xlabel('Price in 000s')
# plt.ylabel('Nr. of Homes')
#
# plt.show()


# Distance to Employment - Length of Commute
# sns.displot(data.DIS,
#             bins=50,
#             aspect=2,
#             kde=True,
#             color='darkblue')
#
# plt.title(f'Distance to Employment Centres. Average: {(data.DIS.mean()):.2}')
# plt.xlabel('Weighted Distance to 5 Boston Employment Centres')
# plt.ylabel('Nr. of Homes')
#
# plt.show()


# Number of Rooms
# sns.displot(data.RM,
#             aspect=2,
#             kde=True,
#             color='#00796b')
#
# plt.title(f'Distribution of Rooms in Boston. Average: {data.RM.mean():.2}')
# plt.xlabel('Average Number of Rooms')
# plt.ylabel('Nr. of Homes')
#
# plt.show()


# Access to Highways
# plt.figure(figsize=(10, 5), dpi=200)
#
# plt.hist(data['RAD'],
#          bins=24,
#          ec='black',
#          color='#7b1fa2',
#          rwidth=0.5)
#
# plt.xlabel('Accessibility to Highways')
# plt.ylabel('Nr. of Houses')
# plt.show()

"""Create a bar chart with plotly for CHAS to show many more homes are away from the river versus next to it."""
river_access = data['CHAS'].value_counts()
# print(river_access)

# bar = px.bar(x=['No', 'Yes'],
#              y=river_access.values,
#              color=river_access.values,
#              color_continuous_scale=px.colors.sequential.haline,
#              title='Next to Charles River?')
#
# bar.update_layout(xaxis_title='Property Located Next to the River?',
#                   yaxis_title='Number of Homes',
#                   coloraxis_showscale=False)
# bar.show()

"""Run a Seaborn .pairplot() to visualise all the relationships at the same time."""
# sns.pairplot(data)
# sns.pairplot(data, kind='reg', plot_kws={'line_kws':{'color': 'cyan'}})
# plt.show()

"""Compare DIS (Distance from employment) with NOX (Nitric Oxide Pollution) using Seaborn's .jointplot()."""
# with sns.axes_style('darkgrid'):
#     sns.jointplot(x=data['DIS'],
#                   y=data['NOX'],
#                   height=8,
#                   color='deeppink',
#                   kind="reg",
#                   joint_kws={'scatter_kws': dict(alpha=0.5)})
#
# plt.show()

"""Compare INDUS (the proportion of non-retail industry i.e., factories) with NOX (Nitric Oxide Pollution) using Seaborn's .jointplot()."""
# with sns.axes_style('darkgrid'):
#     sns.jointplot(x=data.NOX,
#                   y=data.INDUS,
#                   # kind='hex',
#                   height=7,
#                   color='darkgreen',
#                   kind="reg",
#                   joint_kws={'scatter_kws': dict(alpha=0.5)})
#
# plt.show()

"""Compare LSTAT (proportion of lower-income population) with RM (number of rooms) using Seaborn's .jointplot()."""
# with sns.axes_style('darkgrid'):
#     sns.jointplot(x=data['LSTAT'],
#                   y=data['RM'],
#                   # kind='hex',
#                   height=7,
#                   color='orange',
#                   kind="reg",
#                   joint_kws={'scatter_kws': dict(alpha=0.5)})
#
# plt.show()

"""Compare LSTAT with PRICE using Seaborn's .jointplot()."""
# with sns.axes_style('darkgrid'):
#     sns.jointplot(x=data.LSTAT,
#                   y=data.PRICE,
#                   # kind='hex',
#                   height=7,
#                   color='crimson',
#                   kind="reg",
#                   joint_kws={'scatter_kws': dict(alpha=0.5)})
#
# plt.show()

"""Compare RM (number of rooms) with PRICE using Seaborn's .jointplot()."""
# with sns.axes_style('whitegrid'):
#     sns.jointplot(x=data.RM,
#                   y=data.PRICE,
#                   height=7,
#                   color='darkblue',
#                   kind="reg",
#                   joint_kws={'scatter_kws': dict(alpha=0.5)})
#
# plt.show()

"""Split Training & Test Dataset"""
# Step 1: Import the train_test_split() function from sklearn
# Step 2: Create 4 subsets: X_train, X_test, y_train, y_test
# Step 3: Split the training and testing data roughly 80/20.
# Step 4: To get the same random split every time you run your notebook use random_state=10.

# target = data['PRICE']
# features = data.drop('PRICE', axis=1)

target = data.iloc[:, -1]
features = data.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=10)

# % of training set
train_pct = 100*len(X_train)/len(features)
# print(f'Training data is {train_pct:.3}% of the total data.')

# % of test data set
test_pct = 100*X_test.shape[0]/features.shape[0]
# print(f'Test data makes up the remaining {test_pct:0.3}%.')

"""Use sklearn to run the regression on the training dataset. How high is the r-squared for the regression on the training data?"""
# 𝑃𝑅𝐼̂𝐶𝐸 = 𝜃0 + 𝜃1𝑅𝑀 + 𝜃2𝑁𝑂𝑋 + 𝜃3𝐷𝐼𝑆 + 𝜃4𝐶𝐻𝐴𝑆...+ 𝜃13𝐿𝑆𝑇𝐴𝑇
regr = LinearRegression()
regr.fit(X_train, y_train)
rsquared = regr.score(X_train, y_train)

# print(f'Training data r-squared: {rsquared:.2}')

"""Print out the coefficients (the thetas in the equation above) for the features. Hint: You'll see a nice table if you stick the coefficients in a DataFrame."""
# print(regr.coef_)
# print(type(regr.coef_))

regr_coef = pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['Coefficient'])
# print(regr_coef)

"""According to the model, what is the premium you would have to pay for an extra room?"""
# print(regr_coef.loc["RM"]["Coefficient"])
# print(regr_coef.loc["RM"].values[0])

premium = regr_coef.loc['RM'].values[0] * 1000  # i.e., ~3.11 * 1000
# print(f'The price premium for having an extra room is ${premium:.5}')

"""Analyse the Estimated Values & Regression Residuals"""
# How good our regression is depends not only on the r-squared. It also depends on the residuals -
# the difference between the model's predictions ( 𝑦̂𝑖 ) and the true values ( 𝑦𝑖 ) inside y_train.

predicted_values = regr.predict(X_train)
residuals = (y_train - predicted_values)

"""Original Regression of Actual vs. Predicted Prices"""
# The first plot should be actual values (y_train) against the predicted value values:
# The cyan line in the middle shows y_train against y_train. If the predictions had been 100% accurate then all the
# dots would be on this line. The further away the dots are from the line, the worse the prediction was. That makes
# the distance to the cyan line, you guessed it, our residuals.

# plt.figure(dpi=100)
# plt.scatter(x=y_train, y=predicted_values, c='indigo', alpha=0.6)
# plt.plot(y_train, y_train, color='cyan')
# plt.title(f'Actual vs Predicted Prices: $y _i$ vs $\hat y_i$', fontsize=17)
# plt.xlabel('Actual prices 000s $y _i$', fontsize=14)
# plt.ylabel('Prediced prices 000s $\hat y _i$', fontsize=14)
# plt.show()

"""Residuals vs Predicted values"""
# The second plot should be the residuals against the predicted prices.
# The residuals represent the errors of our model. If there's a pattern in our errors, then our model has
# a systematic bias.

# plt.figure(dpi=100)
# plt.scatter(x=predicted_values, y=residuals, c='indigo', alpha=0.6)
# plt.title('Residuals vs Predicted Values', fontsize=17)
# plt.xlabel('Predicted Prices $\hat y _i$', fontsize=14)
# plt.ylabel('Residuals', fontsize=14)
# plt.show()

"""Calculate the mean and the skewness of the residuals. Use Seaborn's .displot() to create a histogram and superimpose the Kernel Density Estimate (KDE)"""
# In an ideal case, we want a normal distribution. A normal distribution has a skewness of 0 and a mean of 0.
# A skew of 0 means that the distribution is symmetrical - the bell curve is not lopsided or biased to one side.

resid_mean = round(residuals.mean(), 2)
resid_skew = round(residuals.skew(), 2)
# print(f'Residuals Skew ({resid_skew}) Mean ({resid_mean})')

# sns.displot(residuals, kde=True, color='indigo')
# plt.title(f'Residuals Skew ({resid_skew}) Mean ({resid_mean})')
# plt.show()

"""Data Transformations for a Better Fit"""
# Investigate if the target data['PRICE'] could be a suitable candidate for a log transformation.
# Step 1: Use Seaborn's .displot() to show a histogram and KDE of the price data.
# Step 2: Calculate the skew of that distribution.
# Step 3: Use NumPy's log() function to create a Series that has the log prices
# Step 4: Plot the log prices using Seaborn's .displot() and calculate the skew.
# Step 5: Compare and see which distribution has a skew that's closer to zero?

# tgt_skew = data['PRICE'].skew()
# sns.displot(data['PRICE'], kde='kde', color='green')
# plt.title(f'Normal Prices. Skew is {tgt_skew:.3}')
# plt.show()

y_log = np.log(data['PRICE'])
# sns.displot(y_log, kde=True)
# plt.title(f'Log Prices. Skew is {y_log.skew():.3}')
# plt.show()

"""How does the log transformation work?"""
# Using a log transformation does not affect every price equally. Large prices are affected more than
# smaller prices in the dataset. We can see this when we plot the actual prices against the (transformed) log prices.

# plt.figure(dpi=150)
# plt.scatter(data.PRICE, np.log(data.PRICE))
#
# plt.title('Mapping the Original Price to a Log Price')
# plt.ylabel('Log Price')
# plt.xlabel('Actual $ Price in 000s')
# plt.show()

"""Regression using Log Prices"""
# log(𝑃𝑅𝐼̂𝐶𝐸) = 𝜃0 + 𝜃1𝑅𝑀 + 𝜃2𝑁𝑂𝑋 + 𝜃3𝐷𝐼𝑆 + 𝜃4𝐶𝐻𝐴𝑆 +...+ 𝜃13𝐿𝑆𝑇𝐴𝑇
# Step 1: Use train_test_split() with the same random state as before to make the results comparable.
# Step 2: Run a second regression, but this time use the transformed target data.
# Step 3: Calculate the r-squared of the regression on the training data

new_target = np.log(data['PRICE']) # Use log prices
features = data.drop('PRICE', axis=1)

X_train, X_test, log_y_train, log_y_test = train_test_split(features,
                                                            new_target,
                                                            test_size=0.2,
                                                            random_state=10)

log_regr = LinearRegression()
log_regr.fit(X_train, log_y_train)
log_rsquared = log_regr.score(X_train, log_y_train)

log_predictions = log_regr.predict(X_train)
log_residuals = (log_y_train - log_predictions)

# print(f'Training data r-squared: {log_rsquared:.2}')

"""Print out the coefficients of the new regression model."""
df_coef = pd.DataFrame(data=log_regr.coef_, index=X_train.columns, columns=['coef'])
# print(df_coef)

"""Regression with Log Prices & Residual Plots"""
# Graph of Actual vs. Predicted Log Prices
# plt.scatter(x=log_y_train, y=log_predictions, c='navy', alpha=0.6)
# plt.plot(log_y_train, log_y_train, color='cyan')
# plt.title(f'Actual vs Predicted Log Prices: $y _i$ vs $\hat y_i$ (R-Squared {log_rsquared:.2})', fontsize=17)
# plt.xlabel('Actual Log Prices $y _i$', fontsize=14)
# plt.ylabel('Prediced Log Prices $\hat y _i$', fontsize=14)
# plt.show()

# Original Regression of Actual vs. Predicted Prices
# plt.scatter(x=y_train, y=predicted_values, c='indigo', alpha=0.6)
# plt.plot(y_train, y_train, color='cyan')
# plt.title(f'Original Actual vs Predicted Prices: $y _i$ vs $\hat y_i$ (R-Squared {rsquared:.3})', fontsize=17)
# plt.xlabel('Actual prices 000s $y _i$', fontsize=14)
# plt.ylabel('Prediced prices 000s $\hat y _i$', fontsize=14)
# plt.show()

# Residuals vs Predicted values (Log prices)
# plt.scatter(x=log_predictions, y=log_residuals, c='navy', alpha=0.6)
# plt.title('Residuals vs Fitted Values for Log Prices', fontsize=17)
# plt.xlabel('Predicted Log Prices $\hat y _i$', fontsize=14)
# plt.ylabel('Residuals', fontsize=14)
# plt.show()

# Residuals vs Predicted values
# plt.scatter(x=predicted_values, y=residuals, c='indigo', alpha=0.6)
# plt.title('Original Residuals vs Fitted Values', fontsize=17)
# plt.xlabel('Predicted Prices $\hat y _i$', fontsize=14)
# plt.ylabel('Residuals', fontsize=14)
# plt.show()

"""Calculate the mean and the skew for the residuals using log prices."""
# Distribution of Residuals (log prices) - checking for normality
log_resid_mean = round(log_residuals.mean(), 2)
log_resid_skew = round(log_residuals.skew(), 2)
# print(f'Residuals Skew ({log_resid_skew}) Mean ({log_resid_mean})')

# sns.displot(log_residuals, kde=True, color='navy')
# plt.title(f'Log price model: Residuals Skew ({log_resid_skew}) Mean ({log_resid_mean})')
# plt.show()

# sns.displot(residuals, kde=True, color='indigo')
# plt.title(f'Original model: Residuals Skew ({resid_skew}) Mean ({resid_mean})')
# plt.show()

"""Compare Out of Sample Performance"""
# The real test is how our model performs on data that it has not "seen" yet. This is where our X_test comes in.
# Compare the r-squared of the two models on the test dataset. Which model does better?

rsquared_test = regr.score(X_test, y_test)
# print(f'Original Model Test Data r-squared: {rsquared_test:.2}')

log_rsquared_test = log_regr.score(X_test, log_y_test)
# print(f'Log Model Test Data r-squared: {log_rsquared_test:.2}')

"""Predict a Property's Value using the Regression Coefficients"""
# log(𝑃𝑅𝐼̂𝐶𝐸) = 𝜃0 + 𝜃1𝑅𝑀 + 𝜃2𝑁𝑂𝑋 + 𝜃3𝐷𝐼𝑆 + 𝜃4𝐶𝐻𝐴𝑆 +...+ 𝜃13𝐿𝑆𝑇𝐴𝑇

# Starting Point: Average Values in the Dataset
features = data.drop(['PRICE'], axis=1)
average_vals = features.mean().values
property_stats = pd.DataFrame(data=average_vals.reshape(1, len(features.columns)), columns=features.columns)
# print(property_stats)

"""Predict how much the average property is worth using the stats above. What is the log price estimate and what is the dollar estimate?"""
# Make prediction
log_estimate = log_regr.predict(property_stats)[0]
# print(f'The log price estimate is ${log_estimate:.3}')

# Convert Log Prices to Acutal Dollar Values
# dollar_est = np.e**log_estimate * 1000
# or use
dollar_est = np.exp(log_estimate) * 1000
# print(f'The property is estimated to be worth ${dollar_est:.6}')

"""Keeping the average values for CRIM, RAD, INDUS and others, value a property with the following characteristics:"""
# Define Property Characteristics
next_to_river = True
nr_rooms = 8
students_per_classroom = 20
distance_to_town = 5
pollution = data.NOX.quantile(q=0.75)  # high
amount_of_poverty = data.LSTAT.quantile(q=0.25)  # low

# Solution:
# Set Property Characteristics
property_stats['RM'] = nr_rooms
property_stats['PTRATIO'] = students_per_classroom
property_stats['DIS'] = distance_to_town

if next_to_river:
    property_stats['CHAS'] = 1
else:
    property_stats['CHAS'] = 0

property_stats['NOX'] = pollution
property_stats['LSTAT'] = amount_of_poverty
# print(property_stats)

# Make prediction
log_estimate = log_regr.predict(property_stats)[0]
print(f'The log price estimate is ${log_estimate:.3}')

# Convert Log Prices to Acutal Dollar Values
dollar_amt = np.e**log_estimate * 1000
print(f'The property is estimated to be worth ${dollar_amt:.6}')

