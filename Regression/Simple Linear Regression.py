
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score


df = pd.read_csv("FuelConsumption.csv")

data = df[['ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]



x = (data.drop(["CO2EMISSIONS"],1)).values
y = (data["CO2EMISSIONS"]).values

# Splitting the data into training and test data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)
print('Train Set', x_train.shape, y_train.shape)
print('Test Set', x_test.shape, y_test.shape)


model = linear_model.LinearRegression()
model.fit(x_train, y_train)

print('Coefficients: ', model.coef_)
print('The Intercept: ', model.intercept_)

# Reshaping y array to fit the line
y = np.repeat(y[..., np.newaxis], 3, -1)

regressionLine = model.coef_*x+model.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, regressionLine)
plt.show()


y_hat = model.predict(x_test)
print('Mean absolute error: %.2f' %np.mean(np.absolute(y_hat - y_test)) )
print('Residual sum of squares(MSE): %.2f' %np.mean((y_hat - y_test)**2))
print('R_2 score: %.2f' %r2_score(y_hat, y_test))

