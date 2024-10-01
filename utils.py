import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load a dataset (example: Iris dataset)
data = load_iris()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (Logistic Regression example)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the trained model to a file using joblib
joblib.dump(model, 'model.pkl')

print("Model saved as 'model.pkl'")
