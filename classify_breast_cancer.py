# An example classification --> Breast Cancer Model
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm):
    """
    This function takes in true labels and predicted labels as NumPy arrays and plots the corresponding confusion matrix using Seaborn.

    Parameters:
        y_true (numpy.ndarray): True labels represented as a NumPy array.
        y_pred (numpy.ndarray): Predicted labels represented as a NumPy array.

    Returns:
        None
    """
    # Set the figure size
    plt.figure(figsize=(8, 6))

    # Create a heatmap using Seaborn
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)

    # Add labels to the x and y axes
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Show the plot
    plt.show()


# Define class labels
class_names = ['Class 0', 'Class 1']

# Load data
data = load_breast_cancer()
X = np.array(pd.DataFrame(data.data, columns=data.feature_names))
Y = np.array(pd.Series(data.target))

# Classify
y1 = Y * 0
y1[np.argwhere(Y == 1)] = 1
y2 = Y * 0
y2[np.argwhere(Y == 0)] = 1
y_new = np.array([y1, y2]).T

state = np.array([0, 1])

# Preprocess data
x_train, x_test, y_train, y_test = train_test_split(X, y_new, test_size=0.2, random_state=1)

# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='sigmoid'),
    tf.keras.layers.Dense(units=2, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, epochs=1000, verbose=0)

# Generate predictions
y_pred = np.round(model.predict(x_test)).astype(int)

# Evaluate model on validation set
cm = confusion_matrix(np.dot(y_test, state), np.dot(y_pred, state))
print('Original Confusion Matrix:\n ', confusion_matrix(np.dot(y_test, state), np.dot(y_test, state)))
print('Model Confusion Matrix:\n', cm)

# Loss
print(f'Loss: {history.history["loss"][-1]}')
loss = history.history['loss']

# MSE
print('MSE:', mse(y_test, y_pred))

# Plot
plt.plot(loss)
plot_confusion_matrix(cm)
