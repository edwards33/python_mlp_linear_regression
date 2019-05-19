import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston_data = datasets.load_boston()

x = boston_data.data
y = boston_data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=16)

regressions_model = LinearRegression()
regressions_model.fit(x_train, y_train)

predictions = regressions_model.predict(x_test)

#r squared, coefficient of determination
print regressions_model.score(x_test, y_test)

#mean squared error
print metrics.mean_squared_error(y_test, predictions)
