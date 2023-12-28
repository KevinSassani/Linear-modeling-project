import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.regressionplots import plot_leverage_resid2, influence_plot
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import qqplot
from gvlma import gvlma

# Load data
diabetes = pd.read_csv("diabetes.txt", sep="\t")

# Convert SEX to categorical
diabetes['SEX'] = diabetes['SEX'].astype('category')

# Display summary statistics
print(diabetes.describe())

# Create dummy variables for SEX
diabetes = pd.get_dummies(diabetes, columns=['SEX'], drop_first=True)

# Display summary statistics again
print(diabetes.describe())

# Display scatter plot matrix
sns.pairplot(diabetes, hue='SEX')
plt.show()

# Partition the data into training and testing sets
X = diabetes.drop('Y', axis=1)
y = diabetes['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2014)

# Ordinary least squares
X_train = sm.add_constant(X_train)  # Add a constant term
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

# Prediction
X_test = sm.add_constant(X_test)
y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Best subset selection
# (Note: This implementation is simplified and may not perfectly match R's leaps function)
from sklearn.feature_selection import RFE

lr = LinearRegression()
selector = RFE(lr, n_features_to_select=1)
selector.fit(X_train, y_train)

best_subset_indices = selector.ranking_
best_subset_indices = np.argsort(best_subset_indices) + 1  # Adjust for 0-based indexing
print("Best Subset Indices:", best_subset_indices)

# Train and test the model on the best subset
model_best_subset = sm.OLS(y_train, X_train.iloc[:, best_subset_indices]).fit()
print(model_best_subset.summary())

y_pred_best_subset = model_best_subset.predict(X_test.iloc[:, best_subset_indices])
print("RMSE (Best Subset):", np.sqrt(mean_squared_error(y_test, y_pred_best_subset)))

# Diagnostics
# Residual plots
residuals = y_train - model.predict(X_train)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(model.fittedvalues, residuals)
plt.title("Residuals vs Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")

# Q-Q plot
plt.subplot(1, 2, 2)
qqplot(residuals, line='s')
plt.title("Q-Q Plot")
plt.show()

# Leverage plot and influence plot
fig, ax = plt.subplots(figsize=(12, 8))
plot_leverage_resid2(model, ax=ax)
influence_plot(model, ax=ax, criterion="cooks")
plt.show()

# Variance inflation factors (VIF)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
print(vif_data)

# Non-constant error variance test
_, p_value, _, _ = het_breuschpagan(model.resid, X_train)
print("Breusch-Pagan p-value:", p_value)

# Autocorrelation test
durbin_watson_statistic = durbin_watson(model.resid)
print("Durbin-Watson Statistic:", durbin_watson_statistic)

# Global test of model assumptions
gvlma_model = gvlma(model)
print(gvlma_model.summary())
