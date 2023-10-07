import numpy as np
from sklearn.linear_model import LinearRegression
# sample script to test the codespace
# Generate the data
X = np.random.rand(100, 2)
y = 2 * X[:, 0] + X[:, 1] + np.random.randn(100)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction
new_data = np.array([[0.5, 0.3]])
prediction = model.predict(new_data)

# Print the prediction
print(f"predicted value for {new_data} is: {prediction[0]}")
