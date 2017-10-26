from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime

import matplotlib.pyplot as plt
import pandas
import skflow
from pylab import savefig
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Load the data set
df = pandas.read_csv("ml_house_data_set.csv")

# Remove the fields from the data set that we don't want to include in our model
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']

# Replace categorical data with one-hot encoded data
features_df = pandas.get_dummies(df, columns=['garage_type', 'city'])
del features_df['sale_price']

X = features_df.as_matrix()
y = df['sale_price'].as_matrix()

# Split the data set in a training set (70%) and a test set (30%)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)

# # Create the model
# model = ensemble.GradientBoostingRegressor()
#
# # Parameters we want to try
# param_grid = {
#     'n_estimators': [500, 1000, 3000],
#     'max_depth': [4, 6],
#     'min_samples_leaf': [3, 5, 9, 17],
#     'learning_rate': [0.1, 0.05, 0.02, 0.01],
#     'max_features': [1.0, 0.3, 0.1],
#     'loss': ['ls', 'lad', 'huber']
# }

# regressor = skflow.TensorFlowDNNRegressor(hidden_units=[10, 10, 10],
#     steps=20000, learning_rate=0.01, batch_size=13)

# regressor = SVR(kernel='rbf', C=1000, gamma='auto')

regressor = skflow.TensorFlowLinearRegressor(steps=2000, learning_rate=0.01, batch_size=13)

ts_a = datetime.datetime.now()

# Train and Predict

regressor.fit(X_train, y_train)
score = metrics.mean_squared_error(regressor.predict(scaler.transform(X_test)), y_test)
X_ty = regressor.predict(X_train)
score1 = metrics.mean_squared_error(X_ty, y_train)

print('Test MSE: {0:f}'.format(score))
print('Train MSE: {0:f}'.format(score1))

# get end timestamp
ts_b = datetime.datetime.now()

delta = ts_b - ts_a

print("Total GPU Runtime Duration was " + repr(int(delta.min)) + " minutes.")

print('Test MSE: {0:f}'.format(score))
print('Train MSE: {0:f}'.format(score1))
########################################################
# Look at the results
fig, ax = plt.subplots()
ax.scatter(y_train, X_ty)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

savefig('result.png')
