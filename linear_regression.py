import numpy as np
from sklearn import linear_model as lm
from sklearn.model_selection import train_test_split as split

# 1. Generate a set of 500 evenly spaced values between -10 and 10.
x = np.linspace(-10, 10, 500)

# 2. Define the function to be learned by the neural network.
y = 5 * x**3 - 2 * x**2 -1 * x + 5

# 3. Split the data into training and testing sets, with a test set size of 20%.
# The input data for the neural network is the transpose of the array [x, x^2, x^3, x^4].
# The output data is the array y.
x_train, x_test, y_train, y_test = split(np.asarray([x, x**2, x**3, x**4]).T, y, test_size=0.2)

# 4. Train a linear regression model on the training data.
model = lm.LinearRegression()
model.fit(x_train, y_train)

# Print the output
print(f'y = {np.round(model.coef_[3])} * x^4 + {np.round(model.coef_[2])} * x^3 + {np.round(model.coef_[1])} * x^2 + {np.round(model.coef_[0])} * x + {np.round(model.intercept_)}')

# 5. Print the coefficient of the highest order term in the polynomial fit.
# This value should be close to zero, indicating that the neural network has learned
# the underlying function without overfitting.
print("Coefficient of x^4 is ", model.coef_[3])

# 6. Calculate the mean squared error of the model on the test set.
# This value should be close to zero, indicating that the neural network has learned
# the underlying function accurately.
mse = np.mean((model.predict(x_test) - y_test)**2)
print("Mean squared error:", mse)
