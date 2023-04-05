import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow as tf
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix as confusion

# import
bc = datasets.load_wine()
X = bc.data;
Y = bc.target;

# classify
y1 = Y*0
y2 = Y*0
y3 = Y*0

y1[np.argwhere(Y==0)] = 1
y2[np.argwhere(Y==1)] = 1
y3[np.argwhere(Y==2)] = 1

Y1 = np.asarray([y1,y2,y3]).T

# split
x_train, x_test, y_train, y_test = split(X, Y1, test_size = 0.1, random_state = 1)

# Layers
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=3))
ann.add(tf.keras.layers.Dense(units=3))

# compile
epoc = 1500
ann.compile(optimizer='adam', loss='mean_squared_error')
v = ann.fit(x_train, y_train, epochs = epoc, verbose = 0)
res = ann.predict(x_test)

loss = v.history['loss']
plt.plot(np.arange(1500), loss)
print("loss = ", loss[-1])
print("mse = ", mse(y_test, np.round(res)))
