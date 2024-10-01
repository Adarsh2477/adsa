from flask import Flask, request, jsonify
import mysql.connector
import joblib  # For loading the ML model

app = Flask(__name__)

# MySQL configuration
mysql_config = {
    'host': 'localhost',        # Use IP address if connecting to a remote MySQL server
    'user': 'root',
    'password': '',             # Replace with your actual MySQL password
    'database': 'mh',
    'auth_plugin': 'mysql_native_password'
}

# Create MySQL connection
db = mysql.connector.connect(**mysql_config)

# Load your pre-trained model (e.g., trained in Scikit-learn or TensorFlow)
model = joblib.load('C:/Users/adarsh/OneDrive/Desktop/chat boat/model.pkl')

# API endpoint to get data and return predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from the request
    feature_1 = data['feature_1']
    feature_2 = data['feature_2']

    # Prepare input for model
    input_data = [[feature_1, feature_2]]

    # Make a prediction using the loaded model
    prediction = model.predict(input_data)[0]

    return jsonify({'prediction': prediction})

# API endpoint to save data into MySQL
@app.route('/save_data', methods=['POST'])
def save_data():
    data = request.json
    feature_1 = data['feature_1']
    feature_2 = data['feature_2']

    cursor = db.cursor()

    # Query to insert data into your table
    query = "INSERT INTO ai (feature_1, feature_2) VALUES (%s, %s)"
    cursor.execute(query, (feature_1, feature_2))
    db.commit()  # Commit the transaction

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
