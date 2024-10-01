# # from flask import Flask, request, jsonify, render_template
# # import mysql.connector

# # app = Flask(__name__)

# # # MySQL connection configuration
# # mysql_config = {
# #     'host': 'localhost',
# #     'user': 'root',
# #     'password': '',
# #     'database': 'demo_db',
# #     'auth_plugin': 'mysql_native_password'
# # }

# # # Connect to MySQL
# # db_connection = mysql.connector.connect(**mysql_config)
# # cursor = db_connection.cursor()

# # # Homepage route to serve HTML form
# # @app.route('/')
# # def index():
# #     return render_template('index.html', chat_history=[])

# # # Endpoint to handle user messages
# # @app.route('/send', methods=['POST'])
# # def handle_message():
# #     user_msg = request.form['msg']
# #     response = ""

# #     # Example: Handle queries related to checkup_records
# #     if 'checkup' in user_msg:
# #         # Extract checkup ID from user message
# #         checkup_id = extract_checkup_id(user_msg)

# #         if checkup_id:
# #             # Query database for height based on checkup ID
# #             cursor.execute("SELECT bmi FROM checkup_form WHERE checkup_id = %s", (checkup_id,))
# #             result = cursor.fetchone()

# #             if result:
# #                 bmi = result[0]  # Assuming 'bmi' field represents height in this example
# #                 response = f"Your height (BMI) for checkup ID {checkup_id} is {bmi}."
# #             else:
# #                 response = f"No height information found for checkup ID {checkup_id}."
# #         else:
# #             response = "Could not extract checkup ID from your query."

# #     elif 'diabetes' in user_msg:
# #         # Query database for count of diabetes patients in training_data table
# #         cursor.execute("SELECT COUNT(*) FROM diabetes_table WHERE outcome = '1'")
# #         result = cursor.fetchone()

# #         if result:
# #             diabetes_count = result[0]
# #             response = f"There are {diabetes_count} patients with diabetes in the training data."
# #         else:
# #             response = "No diabetes patient information found in the training data."

# #     else:
# #         response = "Please specify your query."

# #     return jsonify({'response': response})

# # def extract_checkup_id(user_msg):
# #     # Logic to extract checkup ID from user message
# #     words = user_msg.split()
# #     for word in words:
# #         if word.isdigit():
# #             return int(word)
# #     return None

# # if __name__ == '__main__':
# #     app.run(debug=True)


# # from flask import Flask, request, jsonify, render_template
# # import mysql.connector
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.preprocessing import StandardScaler

# # app = Flask(__name__)

# # # MySQL connection configuration
# # mysql_config = {
# #     'host': 'localhost',
# #     'user': 'root',
# #     'password': '',
# #     'database': 'demo_db',
# #     'auth_plugin': 'mysql_native_password'
# # }

# # # Connect to MySQL
# # db_connection = mysql.connector.connect(**mysql_config)
# # cursor = db_connection.cursor()

# # # Homepage route to serve HTML form
# # @app.route('/')
# # def index():
# #     return render_template('index.html', chat_history=[])

# # # Endpoint to handle user messages
# # @app.route('/send', methods=['POST'])
# # def handle_message():
# #     user_msg = request.form['msg']
# #     response = ""

# #     if 'checkup' in user_msg:
# #         # Extract checkup ID from user message
# #         checkup_id = extract_checkup_id(user_msg)

# #         if checkup_id:
# #             # Query database for height based on checkup ID
# #             cursor.execute("SELECT bmi FROM checkup_form WHERE checkup_id = %s", (checkup_id,))
# #             result = cursor.fetchone()

# #             if result:
# #                 bmi = result[0]  # Assuming 'bmi' field represents height in this example
# #                 response = f"Your height (BMI) for checkup ID {checkup_id} is {bmi}."
# #             else:
# #                 response = f"No height information found for checkup ID {checkup_id}."
# #         else:
# #             response = "Could not extract checkup ID from your query."
            

# #     elif 'diabetes' in user_msg:
# #         if 'prediction' in user_msg:
# #             try:
# #                 # Prediction logic here
# #                 cursor.execute("SELECT * FROM diabetes_table")
# #                 data = cursor.fetchall()
# #                 column_names = [desc[0] for desc in cursor.description]
# #                 df = pd.DataFrame(data, columns=column_names)

# #                 X = df.drop('outcome', axis=1)
# #                 y = df['outcome']

# #                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #                 scaler = StandardScaler()
# #                 X_train_scaled = scaler.fit_transform(X_train)
# #                 X_test_scaled = scaler.transform(X_test)

# #                 model = LogisticRegression()
# #                 model.fit(X_train_scaled, y_train)

# #                 # Example new data for prediction (ensure it has the same number of features as X)
# #                 new_data = np.array([[5, 166, 72, 19, 175, 25.8, 0.587, 51, 1]])  # Replace with actual new data
# #                 new_data_scaled = scaler.transform(new_data)
# #                 prediction = model.predict_proba(new_data_scaled)

# #                 predicted_prob = prediction[0][1]
# #                 response = f"The predicted probability of developing diabetes is {predicted_prob:.2f}."
# #             except ValueError as e:
# #                 response = f"Error: {str(e)}"
# #         else:
# #             cursor.execute("SELECT COUNT(*) FROM diabetes_table WHERE outcome = '1'")
# #             result = cursor.fetchone()

# #             if result:
# #                 diabetes_count = result[0]
# #                 response = f"There are {diabetes_count} patients with diabetes in the training data."
# #             else:
# #                 response = "No diabetes patient information found in the training data."
# #     else:
# #         response = "Please specify your query."

# #     return jsonify({'response': response})

# # def extract_checkup_id(user_msg):
# #     words = user_msg.split()
# #     for word in words:
# #         if word.isdigit():
# #             return int(word)
# #     return None

# # if __name__ == '__main__':
# #     app.run(debug=True)


# # from flask import Flask, request, jsonify, render_template
# # import mysql.connector
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.preprocessing import StandardScaler
# # import spacy

# # app = Flask(__name__)

# # # MySQL connection configuration
# # mysql_config = {
# #     'host': 'localhost',
# #     'user': 'root',
# #     'password': '',
# #     'database': 'demo_db',
# #     'auth_plugin': 'mysql_native_password'
# # }

# # # Connect to MySQL
# # db_connection = mysql.connector.connect(**mysql_config)
# # cursor = db_connection.cursor()

# # # Load English NLP model from spaCy
# # nlp = spacy.load("en_core_web_sm")

# # # Homepage route to serve HTML form
# # @app.route('/')
# # def index():
# #     return render_template('index.html', chat_history=[])

# # # Endpoint to handle user messages
# # @app.route('/send', methods=['POST'])
# # def handle_message():
# #     user_msg = request.form['msg']
# #     response = ""

# #     # Process user message with spaCy
# #     doc = nlp(user_msg.lower())  # Convert to lowercase for consistency

# #     # Extract entities
# #     entities = [ent.text for ent in doc.ents]

# #     # Determine intent based on keywords or entities
# #     if 'checkup' in user_msg:
# #         checkup_id = extract_checkup_id(user_msg)
# #         if checkup_id:
# #             cursor.execute("SELECT bmi FROM checkup_form WHERE checkup_id = %s", (checkup_id,))
# #             result = cursor.fetchone()
# #             if result:
# #                 bmi = result[0]
# #                 response = f"Your height (BMI) for checkup ID {checkup_id} is {bmi}."
# #             else:
# #                 response = f"No height information found for checkup ID {checkup_id}."
# #         else:
# #             response = "Could not extract checkup ID from your query."

# #     elif 'diabetes' in user_msg:
# #         if 'prediction' in user_msg:
# #             try:
# #                 # Prediction logic here
# #                 cursor.execute("SELECT * FROM diabetes_table")
# #                 data = cursor.fetchall()
# #                 column_names = [desc[0] for desc in cursor.description]
# #                 df = pd.DataFrame(data, columns=column_names)

# #                 X = df.drop('outcome', axis=1)
# #                 y = df['outcome']

# #                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #                 scaler = StandardScaler()
# #                 X_train_scaled = scaler.fit_transform(X_train)
# #                 X_test_scaled = scaler.transform(X_test)

# #                 model = LogisticRegression()
# #                 model.fit(X_train_scaled, y_train)

# #                 # Example new data for prediction (ensure it has the same number of features as X)
# #                 new_data = np.array([[5, 166, 72, 19, 175, 25.8, 0.587, 51, 1]])  # Replace with actual new data
# #                 new_data_scaled = scaler.transform(new_data)
# #                 prediction = model.predict_proba(new_data_scaled)

# #                 predicted_prob = prediction[0][1]
# #                 response = f"The predicted probability of developing diabetes is {predicted_prob:.2f}."
# #             except ValueError as e:
# #                 response = f"Error: {str(e)}"
# #         else:
# #             cursor.execute("SELECT COUNT(*) FROM diabetes_table WHERE outcome = '1'")
# #             result = cursor.fetchone()
# #             if result:
# #                 diabetes_count = result[0]
# #                 response = f"There are {diabetes_count} patients with diabetes in the training data."
# #             else:
# #                 response = "No diabetes patient information found in the training data."

# #     else:
# #         response = "Please specify your query."

# #     return jsonify({'response': response})

# # def extract_checkup_id(user_msg):
# #     words = user_msg.split()
# #     for word in words:
# #         if word.isdigit():
# #             return int(word)
# #     return None

# # if __name__ == '__main__':
# #     app.run(debug=True)

# # from flask import Flask, request, jsonify, render_template
# # import mysql.connector
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.preprocessing import StandardScaler
# # import spacy
# # from spellchecker import SpellChecker

# # app = Flask(__name__)

# # # MySQL connection configuration
# # mysql_config = {
# #     'host': 'localhost',
# #     'user': 'root',
# #     'password': '',
# #     'database': 'demo_db',
# #     'auth_plugin': 'mysql_native_password'
# # }

# # # Connect to MySQL
# # db_connection = mysql.connector.connect(**mysql_config)
# # cursor = db_connection.cursor()

# # # Load English NLP model from spaCy
# # nlp = spacy.load("en_core_web_sm")

# # # Initialize SpellChecker
# # spell = SpellChecker()

# # # Homepage route to serve HTML form
# # @app.route('/')
# # def index():
# #     return render_template('index.html', chat_history=[])

# # # Endpoint to handle user messages
# # @app.route('/send', methods=['POST'])
# # def handle_message():
# #     user_msg = request.form['msg']
# #     response = ""

# #     try:
# #         # Correct spelling mistakes
# #         corrected_msg = correct_spelling(user_msg)

# #         # Process corrected user message with spaCy
# #         doc = nlp(corrected_msg.lower())  # Convert to lowercase for consistency

# #         # Extract keywords
# #         keywords = [token.text for token in doc if not token.is_stop]

# #         # Determine intent based on keywords or entities
# #         if any(keyword in keywords for keyword in ['checkup_id', 'bmi', 'height']):
# #             response = handle_checkup(corrected_msg)
# #         elif any(keyword in keywords for keyword in ['diabetes', 'sugar', 'glucose', 'blood sugar']):
# #             if any(keyword in keywords for keyword in ['prediction', 'predict', 'probability', 'chance']):
# #                 response = predict_diabetes()
# #             else:
# #                 response = count_diabetes_patients()
# #         else:
# #             response = "Please specify your query."
# #     except Exception as e:
# #         response = f"An error occurred: {str(e)}"

# #     return jsonify({'response': response})

# # def correct_spelling(user_msg):
# #     try:
# #         if not isinstance(user_msg, str):
# #             raise ValueError(f"Expected a string but got {type(user_msg)}")

# #         words = user_msg.split()
# #         corrected_words = []

# #         for word in words:
# #             corrected_word = spell.correction(word)
# #             if corrected_word is None:
# #                 corrected_word = word
# #             corrected_words.append(corrected_word)
# #             print(f"Original: {word}, Corrected: {corrected_word}")  # Debugging statement

# #         return ' '.join(corrected_words)
# #     except Exception as e:
# #         print(f"Error in correct_spelling: {str(e)}")
# #         return user_msg

# # def handle_checkup(user_msg):
# #     checkup_id = extract_checkup_id(user_msg)
# #     if checkup_id:
# #         cursor.execute("SELECT bmi FROM checkup_form WHERE checkup_id = %s", (checkup_id,))
# #         result = cursor.fetchone()
# #         if result:
# #             bmi = result[0]
# #             return f"Your height (BMI) for checkup ID {checkup_id} is {bmi}."
# #         else:
# #             return f"No height information found for checkup ID {checkup_id}."
# #     else:
# #         return "Could not extract checkup ID from your query."

# # def count_diabetes_patients():
# #     cursor.execute("SELECT COUNT(*) FROM diabetes_table WHERE outcome = '1'")
# #     result = cursor.fetchone()
# #     if result:
# #         diabetes_count = result[0]
# #         return f"There are {diabetes_count} patients with diabetes in the training data."
# #     else:
# #         return "No diabetes patient information found in the training data."

# # def predict_diabetes():
# #     try:
# #         # Fetch data from the database
# #         cursor.execute("SELECT * FROM diabetes_table")
# #         data = cursor.fetchall()
# #         column_names = [desc[0] for desc in cursor.description]
# #         df = pd.DataFrame(data, columns=column_names)

# #         # Preparing the data for training
# #         X = df.drop('outcome', axis=1)
# #         y = df['outcome']
# #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #         scaler = StandardScaler()
# #         X_train_scaled = scaler.fit_transform(X_train)
# #         X_test_scaled = scaler.transform(X_test)

# #         # Train the model
# #         model = LogisticRegression()
# #         model.fit(X_train_scaled, y_train)

# #         # Example new data for prediction (replace with actual new data)
# #         new_data = np.array([[5, 166, 72, 19, 175, 25.8, 0.587, 51, 1]])
# #         new_data_scaled = scaler.transform(new_data)
# #         prediction = model.predict_proba(new_data_scaled)

# #         predicted_prob = prediction[0][1]
# #         return f"The predicted probability of developing diabetes is {predicted_prob:.2f}."
# #     except ValueError as e:
# #         return f"Error: {str(e)}"

# # def extract_checkup_id(user_msg):
# #     words = user_msg.split()
# #     for word in words:
# #         if word.isdigit():
# #             return int(word)
# #     return None

# # if __name__ == '__main__':
# #     app.run(debug=True)






# #  def predict_future_diabetes():
# #     try:
# #         # Fetch employee data from the database
# #         cursor.execute("SELECT * FROM diabetes_table")
# #         data = cursor.fetchall()
# #         column_names = [desc[0] for desc in cursor.description]
# #         df = pd.DataFrame(data, columns=column_names)

# #         # Ensure consistent feature set
# #         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']

# #         # Preparing the data for training
# #         X = df[feature_columns]
# #         y = df['outcome']
# #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #         scaler = StandardScaler()
# #         X_train_scaled = scaler.fit_transform(X_train)
# #         X_test_scaled = scaler.transform(X_test)

# #         # Train the model
# #         model = LogisticRegression()
# #         model.fit(X_train_scaled, y_train)

# #         # Predict for all employees
# #         if 'emp_id' in df.columns:
# #             employee_ids = df['emp_id'].unique()
# #             predictions = []

# #             for emp_id in employee_ids:
# #                 emp_data = df[df['emp_id'] == emp_id][feature_columns]
# #                 emp_data_scaled = scaler.transform(emp_data)
# #                 prediction = model.predict_proba(emp_data_scaled)
# #                 predicted_prob = prediction[:, 1].mean()  # Average probability across years
# #                 predictions.append((emp_id, predicted_prob))

# #             threshold = 0.5  # Adjust threshold as needed
# #             high_risk_employees = [emp_id for emp_id, prob in predictions if prob > threshold]

# #             # Format response
# #             response = "Employees with high risk of developing diabetes:\n"
# #             for emp_id in high_risk_employees:
# #                 response += f"Employee ID {emp_id}\n"

# #             return response
# #         else:
# #             return "Employee ID information is not available in the data."
# #     except Exception as e:
# #         return f"Error: {str(e)}"

# # def predict_future_diabetes():
# #     try:
# #         # Fetch employee data from the database
# #         cursor.execute("SELECT * FROM diabetes_table")
# #         data = cursor.fetchall()
# #         column_names = [desc[0] for desc in cursor.description]
# #         df = pd.DataFrame(data, columns=column_names)

# #         # Ensure consistent feature set
# #         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']

# #         # Preparing the data for training
# #         X = df[feature_columns]
# #         y = df['outcome']
# #         scaler = StandardScaler()
# #         X_scaled = scaler.fit_transform(X)

# #         # Train the model
# #         model = LogisticRegression()
# #         model.fit(X_scaled, y)

# #         # Predict for all employees
# #         if 'emp_id' in df.columns:
# #             employee_insights = []

# #             for emp_id in df['emp_id'].unique():
# #                 emp_data = df[df['emp_id'] == emp_id][feature_columns]
# #                 emp_data_scaled = scaler.transform(emp_data)
# #                 prediction = model.predict_proba(emp_data_scaled)
# #                 predicted_prob = prediction[:, 1].mean()  # Average probability across all data

# #                 # Classify based on predicted probability
# #                 if predicted_prob > 0.5:
# #                     risk_status = "High risk"
# #                 elif predicted_prob > 0.2:  # Adjust thresholds as needed
# #                     risk_status = "Moderate risk"
# #                 else:
# #                     risk_status = "Low risk"

# #                 employee_insights.append((emp_id, predicted_prob, risk_status))

# #             # Format response
# #             if employee_insights:
# #                 response = "Diabetes risk insights for employees:\n"
# #                 for emp_id, prob, risk_status in employee_insights:
# #                     response += f"Employee ID {emp_id}: Predicted probability - {prob:.2f}, Status - {risk_status}\n"
# #             else:
# #                 response = "No employee data found."

# #             return response
# #         else:
# #             return "Employee ID information is not available in the data."
# #     except Exception as e:
# #         return f"Error: {str(e)}"


# # def predict_future_diabetes():
# #     try:
# #         # Fetch employee data from the database
# #         cursor.execute("SELECT * FROM diabetes_table")
# #         data = cursor.fetchall()
# #         column_names = [desc[0] for desc in cursor.description]
# #         df = pd.DataFrame(data, columns=column_names)

# #         # Ensure consistent feature set
# #         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']

# #         # Preparing the data for training
# #         X = df[feature_columns]
# #         y = df['outcome']
# #         scaler = StandardScaler()
# #         X_scaled = scaler.fit_transform(X)

# #         # Train the model
# #         model = LogisticRegression()
# #         model.fit(X_scaled, y)

# #         # Predict for all employees
# #         if 'emp_id' in df.columns:
# #             employee_insights = []

# #             for emp_id in df['emp_id'].unique():
# #                 emp_data = df[df['emp_id'] == emp_id][feature_columns]
# #                 emp_data_scaled = scaler.transform(emp_data)
# #                 prediction = model.predict_proba(emp_data_scaled)
# #                 predicted_prob = prediction[:, 1].mean()  # Average probability across all data

# #                 # Classify based on predicted probability
# #                 if predicted_prob > 0.5:
# #                     risk_status = "High risk"
# #                     future_diabetes_patients = "Future diabetes patients"
# #                 elif predicted_prob > 0.2:  # Adjust thresholds as needed
# #                     risk_status = "Moderate risk"
# #                     future_diabetes_patients = "Future diabetes patients"
# #                 else:
# #                     risk_status = "Low risk"
# #                     future_diabetes_patients = "Not future diabetes patients"

# #                 employee_insights.append((emp_id, predicted_prob, risk_status, future_diabetes_patients))

# #             # Format response
# #             if employee_insights:
# #                 response = "Diabetes risk insights for employees:\n"
# #                 for emp_id, prob, risk_status, future_status in employee_insights:
# #                     response += f"Employee ID {emp_id}: Predicted probability - {prob:.2f}, Status - {risk_status}, {future_status}\n"
# #             else:
# #                 response = "No employee data found."

# #             return response
# #         else:
# #             return "Employee ID information is not available in the data."
# #     except Exception as e:
# #         return f"Error: {str(e)}"




# # from flask import Flask, request, jsonify, render_template
# # import mysql.connector
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.preprocessing import StandardScaler
# # import spacy
# # from spellchecker import SpellChecker

# # app = Flask(__name__)

# # # MySQL connection configuration
# # mysql_config = {
# #     'host': 'localhost',
# #     'user': 'root',
# #     'password': '',
# #     'database': 'demo_db',
# #     'auth_plugin': 'mysql_native_password'
# # }

# # # Connect to MySQL
# # db_connection = mysql.connector.connect(**mysql_config)
# # cursor = db_connection.cursor()

# # # Load English NLP model from spaCy
# # nlp = spacy.load("en_core_web_sm")

# # # Initialize SpellChecker
# # spell = SpellChecker()

# # # Homepage route to serve HTML form
# # @app.route('/')
# # def index():
# #     return render_template('index.html', chat_history=[])

# # # Endpoint to handle user messages
# # @app.route('/send', methods=['POST'])
# # def handle_message():
# #     user_msg = request.form['msg']
# #     response = ""

# #     try:
# #         # Correct spelling mistakes
# #         corrected_msg = correct_spelling(user_msg)

# #         # Process corrected user message with spaCy
# #         doc = nlp(corrected_msg.lower())  # Convert to lowercase for consistency

# #         # Extract keywords
# #         keywords = [token.text for token in doc if not token.is_stop]

# #         # Determine intent based on keywords or entities
# #         if any(keyword in keywords for keyword in ['checkup_id', 'bmi', 'height']):
# #             response = handle_checkup(corrected_msg)
# #         elif any(keyword in keywords for keyword in ['diabetes', 'sugar', 'glucose', 'blood sugar']):
# #             if any(keyword in keywords for keyword in ['prediction', 'predict', 'probability', 'chance']):
# #                 response = predict_diabetes()
# #             else:
# #                 response = count_diabetes_patients()
# #         elif any(keyword in keywords for keyword in ['employee', 'emp', 'future', 'prediction', 'predict']):
# #             response = predict_future_diabetes()
# #         else:
# #             response = "Please specify your query."
            
# #     except Exception as e:
# #         response = f"An error occurred: {str(e)}"

# #     return jsonify({'response': response})

# # def correct_spelling(user_msg):
# #     try:
# #         if not isinstance(user_msg, str):
# #             raise ValueError(f"Expected a string but got {type(user_msg)}")

# #         words = user_msg.split()
# #         corrected_words = []

# #         for word in words:
# #             corrected_word = spell.correction(word)
# #             if corrected_word is None:
# #                 corrected_word = word
# #             corrected_words.append(corrected_word)
# #             print(f"Original: {word}, Corrected: {corrected_word}")  # Debugging statement

# #         return ' '.join(corrected_words)
# #     except Exception as e:
# #         print(f"Error in correct_spelling: {str(e)}")
# #         return user_msg

# # def handle_checkup(user_msg):
# #     checkup_id = extract_checkup_id(user_msg)
# #     if checkup_id:
# #         cursor.execute("SELECT bmi FROM checkup_form WHERE checkup_id = %s", (checkup_id,))
# #         result = cursor.fetchone()
# #         if result:
# #             bmi = result[0]
# #             return f"Your height (BMI) for checkup ID {checkup_id} is {bmi}."
# #         else:
# #             return f"No height information found for checkup ID {checkup_id}."
# #     else:
# #         return "Could not extract checkup ID from your query."

# # def count_diabetes_patients():
# #     cursor.execute("SELECT COUNT(*) FROM diabetes_table WHERE outcome = '1'")
# #     result = cursor.fetchone()
# #     if result:
# #         diabetes_count = result[0]
# #         return f"There are {diabetes_count} patients with diabetes in the training data."
# #     else:
# #         return "No diabetes patient information found in the training data."

# # def predict_diabetes():
# #     try:
# #         # Fetch data from the database
# #         cursor.execute("SELECT * FROM diabetes_table")
# #         data = cursor.fetchall()
# #         column_names = [desc[0] for desc in cursor.description]
# #         df = pd.DataFrame(data, columns=column_names) # Added print statement

# #         # Ensure consistent feature set
# #         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']

# #         # Preparing the data for training
# #         X = df[feature_columns]
# #         y = df['outcome']
# #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #         scaler = StandardScaler()
# #         X_train_scaled = scaler.fit_transform(X_train)
# #         X_test_scaled = scaler.transform(X_test)

# #         # Train the model
# #         model = LogisticRegression()
# #         model.fit(X_train_scaled, y_train)

# #         # Example new data for prediction (replace with actual new data)
# #         new_data = np.array([[5, 166, 72, 19, 175, 25.8, 0.587, 51]])
# #         new_data_scaled = scaler.transform(new_data)
# #         prediction = model.predict_proba(new_data_scaled)

# #         predicted_prob = prediction[0][1]
# #         return f"The predicted probability of developing diabetes is {predicted_prob:.2f}."
# #     except ValueError as e:
# #         return f"Error: {str(e)}"

# # def predict_future_diabetes():
# #     try:
# #         # Fetch employee data from the database
# #         cursor.execute("SELECT * FROM diabetes_table")
# #         data = cursor.fetchall()
# #         column_names = [desc[0] for desc in cursor.description]
# #         df = pd.DataFrame(data, columns=column_names)

# #         # Ensure consistent feature set
# #         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']

# #         # Preparing the data for training
# #         X = df[feature_columns]
# #         y = df['outcome']
# #         scaler = StandardScaler()
# #         X_scaled = scaler.fit_transform(X)

# #         # Train the model
# #         model = LogisticRegression()
# #         model.fit(X_scaled, y)

# #         # Predict for all employees
# #         if 'emp_id' in df.columns:
# #             employee_insights = []
# #             high_risk_count = 0
# #             moderate_risk_count = 0
# #             low_risk_count = 0

# #             for emp_id in df['emp_id'].unique():
# #                 emp_data = df[df['emp_id'] == emp_id][feature_columns]
# #                 emp_data_scaled = scaler.transform(emp_data)
# #                 prediction = model.predict_proba(emp_data_scaled)
# #                 predicted_prob = prediction[:, 1].mean()  # Average probability across all data

# #                 # Classify based on predicted probability
# #                 if predicted_prob > 0.5:
# #                     risk_status = "High risk"
# #                     future_diabetes_patients = "Future diabetes patients"
# #                     high_risk_count += 1
# #                 elif predicted_prob > 0.2:  # Adjust thresholds as needed
# #                     risk_status = "Moderate risk"
# #                     future_diabetes_patients = "Future diabetes patients"
# #                     moderate_risk_count += 1
# #                 else:
# #                     risk_status = "Low risk"
# #                     future_diabetes_patients = "Not future diabetes patients"
# #                     low_risk_count += 1

# #                 employee_insights.append((emp_id, predicted_prob, risk_status, future_diabetes_patients))

# #             # Format response
# #             if employee_insights:
# #                 response = "Diabetes risk insights for employees:\n"
# #                 for emp_id, prob, risk_status, future_status in employee_insights:
# #                     response += f"Employee ID {emp_id}: Predicted probability - {prob:.2f}, Status - {risk_status}, {future_status}\n"
# #                 response += f"\nHigh Risk Count: {high_risk_count}\nModerate Risk Count: {moderate_risk_count}\nLow Risk Count: {low_risk_count}"
# #             else:
# #                 response = "No employee data found."

# #             return response
# #         else:
# #             return "Employee ID information is not available in the data."
# #     except Exception as e:
# #         return f"Error: {str(e)}"


# # def extract_checkup_id(user_msg):
# #     words = user_msg.split()
# #     for word in words:
# #         if word.isdigit():
# #             return int(word)
# #     return None

# # if __name__ == '__main__':
# #     app.run(debug=True)


# # from flask import Flask, request, jsonify, render_template
# # import mysql.connector
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.preprocessing import StandardScaler
# # import spacy
# # from spellchecker import SpellChecker

# # app = Flask(__name__)

# # # MySQL connection configuration
# # mysql_config = {
# #     'host': 'localhost',
# #     'user': 'root',
# #     'password': '',
# #     'database': 'demo_db',
# #     'auth_plugin': 'mysql_native_password'
# # }

# # # Connect to MySQL
# # db_connection = mysql.connector.connect(**mysql_config)
# # cursor = db_connection.cursor()

# # # Load English NLP model from spaCy
# # nlp = spacy.load("en_core_web_sm")

# # # Homepage route to serve HTML form
# # @app.route('/')
# # def index():
# #     return render_template('index.html', chat_history=[])

# # # Function to handle checkup queries
# # def handle_checkup(user_msg):
# #     checkup_id = extract_checkup_id(user_msg)
# #     if checkup_id:
# #         cursor.execute("SELECT bmi FROM checkup_form WHERE checkup_id = %s", (checkup_id,))
# #         result = cursor.fetchone()
# #         if result:
# #             bmi = result[0]
# #             return f"Your height (BMI) for checkup ID {checkup_id} is {bmi}."
# #         else:
# #             return f"No height information found for checkup ID {checkup_id}."
# #     else:
# #         return "Could not extract checkup ID from your query."

# # # Function to count diabetes patients
# # def count_diabetes_patients(user_msg):
# #     cursor.execute("SELECT COUNT(*) FROM diabetes_table WHERE outcome = '1'")
# #     result = cursor.fetchone()
# #     if result:
# #         diabetes_count = result[0]
# #         return f"There are {diabetes_count} patients with diabetes in the training data."
# #     else:
# #         return "No diabetes patient information found in the training data."

# # def predict_diabetes(user_msg):
# #     try:
# #         # Fetch data from the database
# #         cursor.execute("SELECT * FROM diabetes_table")
# #         data = cursor.fetchall()
# #         column_names = [desc[0] for desc in cursor.description]
# #         df = pd.DataFrame(data, columns=column_names) # Added print statement

# #         # Ensure consistent feature set
# #         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']

# #         # Preparing the data for training
# #         X = df[feature_columns]
# #         y = df['outcome']
# #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #         scaler = StandardScaler()
# #         X_train_scaled = scaler.fit_transform(X_train)
# #         X_test_scaled = scaler.transform(X_test)

# #         # Train the model
# #         model = LogisticRegression()
# #         model.fit(X_train_scaled, y_train)

# #         # Example new data for prediction (replace with actual new data)
# #         new_data = np.array([[5, 166, 72, 19, 175, 25.8, 0.587, 51]])
# #         new_data_scaled = scaler.transform(new_data)
# #         prediction = model.predict_proba(new_data_scaled)

# #         predicted_prob = prediction[0][1]
# #         return f"The predicted probability of developing diabetes is {predicted_prob:.2f}."
# #     except ValueError as e:
# #         return f"Error: {str(e)}"

# # def predict_future_diabetes(user_msg):
# #     try:
# #         # Fetch employee data from the database
# #         cursor.execute("SELECT * FROM diabetes_table")
# #         data = cursor.fetchall()
# #         column_names = [desc[0] for desc in cursor.description]
# #         df = pd.DataFrame(data, columns=column_names)

# #         # Ensure consistent feature set
# #         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']

# #         # Preparing the data for training
# #         X = df[feature_columns]
# #         y = df['outcome']
# #         scaler = StandardScaler()
# #         X_scaled = scaler.fit_transform(X)

# #         # Train the model
# #         model = LogisticRegression()
# #         model.fit(X_scaled, y)

# #         # Predict for all employees
# #         if 'emp_id' in df.columns:
# #             employee_insights = []
# #             high_risk_count = 0
# #             moderate_risk_count = 0
# #             low_risk_count = 0

# #             for emp_id in df['emp_id'].unique():
# #                 emp_data = df[df['emp_id'] == emp_id][feature_columns]
# #                 emp_data_scaled = scaler.transform(emp_data)
# #                 prediction = model.predict_proba(emp_data_scaled)
# #                 predicted_prob = prediction[:, 1].mean()  # Average probability across all data

# #                 # Classify based on predicted probability
# #                 if predicted_prob > 0.5:
# #                     risk_status = "High risk"
# #                     future_diabetes_patients = "Future diabetes patients"
# #                     high_risk_count += 1
# #                 elif predicted_prob > 0.2:  # Adjust thresholds as needed
# #                     risk_status = "Moderate risk"
# #                     future_diabetes_patients = "Future diabetes patients"
# #                     moderate_risk_count += 1
# #                 else:
# #                     risk_status = "Low risk"
# #                     future_diabetes_patients = "Not future diabetes patients"
# #                     low_risk_count += 1

# #                 employee_insights.append((emp_id, predicted_prob, risk_status, future_diabetes_patients))

# #             # Format response
# #             if employee_insights:
# #                 response = "Diabetes risk insights for employees:\n"
# #                 for emp_id, prob, risk_status, future_status in employee_insights:
# #                     response += f"Employee ID {emp_id}: Predicted probability - {prob:.2f}, Status - {risk_status}, {future_status}\n"
# #                 response += f"\nHigh Risk Count: {high_risk_count}\nModerate Risk Count: {moderate_risk_count}\nLow Risk Count: {low_risk_count}"
# #             else:
# #                 response = "No employee data found."

# #             return response
# #         else:
# #             return "Employee ID information is not available in the data."
# #     except Exception as e:
# #         return f"Error: {str(e)}"


# # # Function to predict diabetes (dummy function for example)
# # # def predict_diabetes(user_msg):
# # #     # Placeholder for diabetes prediction logic
# # #     return "Diabetes prediction feature is under development."

# # # # Function to predict future diabetes (dummy function for example)
# # # def predict_future_diabetes(user_msg):
# # #     # Placeholder for future diabetes prediction logic
# # #     return "Future diabetes prediction feature is under development."

# # # Function to extract checkup ID from user message
# # def extract_checkup_id(user_msg):
# #     words = user_msg.split()
# #     for word in words:
# #         if word.isdigit():
# #             return int(word)
# #     return None

# # # Endpoint to handle user messages
# # @app.route('/send', methods=['POST'])
# # def handle_message():
# #     user_msg = request.form['msg'].lower()  # Convert to lowercase for consistency
# #     response = ""

# #     # Use NLP to extract tokens from the user's message
# #     doc = nlp(user_msg)
# #     tokens = [token.text for token in doc]

# #     # Check if any keyword in the user's message matches a keyword in the database
# #     for token in tokens:
# #         cursor.execute("SELECT function_name FROM keyword_functions WHERE keyword LIKE %s", (f"%{token}%",))
# #         result = cursor.fetchone()
# #         if result:
# #             function_name = result[0]
# #             if function_name in globals():
# #                 response = globals()[function_name](user_msg)
# #             else:
# #                 response = "Function not implemented."
# #             break
# #     else:
# #         response = "Keyword not recognized. Please try another query."

# #     return jsonify({'response': response})

# # if __name__ == '__main__':
# #     app.run(debug=True)


# # from flask import Flask, request, jsonify, render_template
# # import mysql.connector
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.preprocessing import StandardScaler
# # import spacy
# # from spellchecker import SpellChecker

# # app = Flask(__name__)

# # # MySQL connection configuration
# # mysql_config = {
# #     'host': 'localhost',
# #     'user': 'root',
# #     'password': '',
# #     'database': 'demo_db',
# #     'auth_plugin': 'mysql_native_password'
# # }

# # # Connect to MySQL
# # db_connection = mysql.connector.connect(**mysql_config)
# # cursor = db_connection.cursor()

# # # Load English NLP model from spaCy
# # nlp = spacy.load("en_core_web_sm")

# # # Homepage route to serve HTML form
# # @app.route('/')
# # def index():
# #     return render_template('index.html', chat_history=[])

# # # Function to handle checkup queries
# # def handle_checkup(user_msg):
# #     checkup_id = extract_checkup_id(user_msg)
# #     if checkup_id:
# #         cursor.execute("SELECT bmi FROM checkup_form WHERE checkup_id = %s", (checkup_id,))
# #         result = cursor.fetchone()
# #         if result:
# #             bmi = result[0]
# #             return f"Your height (BMI) for checkup ID {checkup_id} is {bmi}."
# #         else:
# #             return f"No height information found for checkup ID {checkup_id}."
# #     else:
# #         return "Could not extract checkup ID from your query."

# # # Function to count diabetes patients
# # def count_diabetes_patients(user_msg):
# #     cursor.execute("SELECT COUNT(*) FROM diabetes_table WHERE outcome = '1'")
# #     result = cursor.fetchone()
# #     if result:
# #         diabetes_count = result[0]
# #         return f"There are {diabetes_count} patients with diabetes in the training data."
# #     else:
# #         return "No diabetes patient information found in the training data."

# # def predict_diabetes(user_msg):
# #     try:
# #         # Fetch data from the database
# #         cursor.execute("SELECT * FROM diabetes_table")
# #         data = cursor.fetchall()
# #         column_names = [desc[0] for desc in cursor.description]
# #         df = pd.DataFrame(data, columns=column_names) # Added print statement

# #         # Ensure consistent feature set
# #         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']

# #         # Preparing the data for training
# #         X = df[feature_columns]
# #         y = df['outcome']
# #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #         scaler = StandardScaler()
# #         X_train_scaled = scaler.fit_transform(X_train)
# #         X_test_scaled = scaler.transform(X_test)

# #         # Train the model
# #         model = LogisticRegression()
# #         model.fit(X_train_scaled, y_train)

# #         # Example new data for prediction (replace with actual new data)
# #         new_data = np.array([[5, 166, 72, 19, 175, 25.8, 0.587, 51]])
# #         new_data_scaled = scaler.transform(new_data)
# #         prediction = model.predict_proba(new_data_scaled)

# #         predicted_prob = prediction[0][1]
# #         return f"The predicted probability of developing diabetes is {predicted_prob:.2f}."
# #     except ValueError as e:
# #         return f"Error: {str(e)}"

# # def predict_future_diabetes(user_msg):
# #     try:
# #         # Fetch employee data from the database
# #         cursor.execute("SELECT * FROM diabetes_table")
# #         data = cursor.fetchall()
# #         column_names = [desc[0] for desc in cursor.description]
# #         df = pd.DataFrame(data, columns=column_names)

# #         # Ensure consistent feature set
# #         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']

# #         # Preparing the data for training
# #         X = df[feature_columns]
# #         y = df['outcome']
# #         scaler = StandardScaler()
# #         X_scaled = scaler.fit_transform(X)

# #         # Train the model
# #         model = LogisticRegression()
# #         model.fit(X_scaled, y)

# #         # Predict for all employees
# #         if 'emp_id' in df.columns:
# #             employee_insights = []
# #             high_risk_count = 0
# #             moderate_risk_count = 0
# #             low_risk_count = 0

# #             for emp_id in df['emp_id'].unique():
# #                 emp_data = df[df['emp_id'] == emp_id][feature_columns]
# #                 emp_data_scaled = scaler.transform(emp_data)
# #                 prediction = model.predict_proba(emp_data_scaled)
# #                 predicted_prob = prediction[:, 1].mean()  # Average probability across all data

# #                 # Classify based on predicted probability
# #                 if predicted_prob > 0.5:
# #                     risk_status = "High risk"
# #                     future_diabetes_patients = "Future diabetes patients"
# #                     high_risk_count += 1
# #                 elif predicted_prob > 0.2:  # Adjust thresholds as needed
# #                     risk_status = "Moderate risk"
# #                     future_diabetes_patients = "Future diabetes patients"
# #                     moderate_risk_count += 1
# #                 else:
# #                     risk_status = "Low risk"
# #                     future_diabetes_patients = "Not future diabetes patients"
# #                     low_risk_count += 1

# #                 employee_insights.append((emp_id, predicted_prob, risk_status, future_diabetes_patients))

# #             # Format response
# #             if employee_insights:
# #                 response = "Diabetes risk insights for employees:\n"
# #                 for emp_id, prob, risk_status, future_status in employee_insights:
# #                     response += f"Employee ID {emp_id}: Predicted probability - {prob:.2f}, Status - {risk_status}, {future_status}\n"
# #                 response += f"\nHigh Risk Count: {high_risk_count}\nModerate Risk Count: {moderate_risk_count}\nLow Risk Count: {low_risk_count}"
# #             else:
# #                 response = "No employee data found."

# #             return response
# #         else:
# #             return "Employee ID information is not available in the data."
# #     except Exception as e:
# #         return f"Error: {str(e)}"


# # # Function to predict diabetes (dummy function for example)
# # # def predict_diabetes(user_msg):
# # #     # Placeholder for diabetes prediction logic
# # #     return "Diabetes prediction feature is under development."

# # # # Function to predict future diabetes (dummy function for example)
# # # def predict_future_diabetes(user_msg):
# # #     # Placeholder for future diabetes prediction logic
# # #     return "Future diabetes prediction feature is under development."

# # # Function to extract checkup ID from user message
# # def extract_checkup_id(user_msg):
# #     words = user_msg.split()
# #     for word in words:
# #         if word.isdigit():
# #             return int(word)
# #     return None

# # # Endpoint to handle user messages
# # @app.route('/send', methods=['POST'])
# # def handle_message():
# #     user_msg = request.form['msg'].lower()  # Convert to lowercase for consistency
# #     response = ""

# #     # Use NLP to extract tokens from the user's message
# #     doc = nlp(user_msg)
# #     tokens = [token.text for token in doc]

# #     # Check if any keyword in the user's message matches a keyword in the database
# #     for token in tokens:
# #         cursor.execute("SELECT function_name FROM keyword_functions WHERE keyword LIKE %s", (f"%{token}%",))
# #         result = cursor.fetchone()
# #         if result:
# #             function_name = result[0]
# #             if function_name in globals():
# #                 response = globals()[function_name](user_msg)
# #             else:
# #                 response = "Function not implemented."
# #             break
# #     else:
# #         response = "Keyword not recognized. Please try another query."

# #     return jsonify({'response': response})

# # if __name__ == '__main__':
# #     app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# # import mysql.connector
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.preprocessing import StandardScaler
# # import spacy
# # from spellchecker import SpellChecker

# # app = Flask(__name__)

# # # MySQL connection configuration
# # mysql_config = {
# #     'host': 'localhost',
# #     'user': 'root',
# #     'password': '',
# #     'database': 'demo_db',
# #     'auth_plugin': 'mysql_native_password'
# # }

# # # Connect to MySQL
# # db_connection = mysql.connector.connect(**mysql_config)
# # cursor = db_connection.cursor()

# # # Load English NLP model from spaCy
# # nlp = spacy.load("en_core_web_sm")

# # # Homepage route to serve HTML form
# # @app.route('/')
# # def index():
# #     return render_template('index.html', chat_history=[])

# # # Function to handle checkup queries
# # def handle_checkup(user_msg):
# #     checkup_id = extract_checkup_id(user_msg)
# #     if checkup_id:
# #         cursor.execute("SELECT bmi FROM checkup_form WHERE checkup_id = %s", (checkup_id,))
# #         result = cursor.fetchone()
# #         if result:
# #             bmi = result[0]
# #             return f"Your height (BMI) for checkup ID {checkup_id} is {bmi}."
# #         else:
# #             return f"No height information found for checkup ID {checkup_id}."
# #     else:
# #         return "Could not extract checkup ID from your query."

# # # Function to count diabetes patients
# # def count_diabetes_patients(user_msg):
# #     cursor.execute("SELECT COUNT(*) FROM diabetes_table WHERE outcome = '1'")
# #     result = cursor.fetchone()
# #     if result:
# #         diabetes_count = result[0]
# #         return f"There are {diabetes_count} patients with diabetes in the training data."
# #     else:
# #         return "No diabetes patient information found in the training data."

# # def predict_diabetes(user_msg):
# #     try:
# #         # Fetch data from the database
# #         cursor.execute("SELECT * FROM diabetes_table")
# #         data = cursor.fetchall()
# #         column_names = [desc[0] for desc in cursor.description]
# #         df = pd.DataFrame(data, columns=column_names)

# #         # Ensure consistent feature set
# #         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']

# #         # Preparing the data for training
# #         X = df[feature_columns]
# #         y = df['outcome']
# #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #         scaler = StandardScaler()
# #         X_train_scaled = scaler.fit_transform(X_train)
# #         X_test_scaled = scaler.transform(X_test)

# #         # Train the model
# #         model = LogisticRegression()
# #         model.fit(X_train_scaled, y_train)

# #         # Example new data for prediction (replace with actual new data)
# #         new_data_dict = {
# #             'pregnancies': [5], 
# #             'glucose': [166], 
# #             'blood_pressure': [72], 
# #             'skin_thickness': [19], 
# #             'insulin': [175], 
# #             'bmi': [25.8], 
# #             'diabetes_pedigree_function': [0.587], 
# #             'age': [51]
# #         }
# #         new_data_df = pd.DataFrame(new_data_dict)
# #         new_data_scaled = scaler.transform(new_data_df)
# #         prediction = model.predict_proba(new_data_scaled)

# #         predicted_prob = prediction[0][1]
# #         return f"The predicted probability of developing diabetes is {predicted_prob:.2f}."
# #     except ValueError as e:
# #         return f"Error: {str(e)}"

# # def predict_future_diabetes(user_msg):
# #     try:
# #         # Fetch employee data from the database
# #         cursor.execute("SELECT * FROM diabetes_table")
# #         data = cursor.fetchall()
# #         column_names = [desc[0] for desc in cursor.description]
# #         df = pd.DataFrame(data, columns=column_names)

# #         # Ensure consistent feature set
# #         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']

# #         # Preparing the data for training
# #         X = df[feature_columns]
# #         y = df['outcome']
# #         scaler = StandardScaler()
# #         X_scaled = scaler.fit_transform(X)

# #         # Train the model
# #         model = LogisticRegression()
# #         model.fit(X_scaled, y)

# #         # Predict for all employees
# #         if 'emp_id' in df.columns:
# #             employee_insights = []
# #             high_risk_count = 0
# #             moderate_risk_count = 0
# #             low_risk_count = 0

# #             for emp_id in df['emp_id'].unique():
# #                 emp_data = df[df['emp_id'] == emp_id][feature_columns]
# #                 emp_data_scaled = scaler.transform(emp_data)
# #                 prediction = model.predict_proba(emp_data_scaled)
# #                 predicted_prob = prediction[:, 1].mean()  # Average probability across all data

# #                 # Classify based on predicted probability
# #                 if predicted_prob > 0.5:
# #                     risk_status = "High risk"
# #                     future_diabetes_patients = "Future diabetes patients"
# #                     high_risk_count += 1
# #                 elif predicted_prob > 0.2:  # Adjust thresholds as needed
# #                     risk_status = "Moderate risk"
# #                     future_diabetes_patients = "Future diabetes patients"
# #                     moderate_risk_count += 1
# #                 else:
# #                     risk_status = "Low risk"
# #                     future_diabetes_patients = "Not future diabetes patients"
# #                     low_risk_count += 1

# #                 employee_insights.append((emp_id, predicted_prob, risk_status, future_diabetes_patients))

# #             # Format response
# #             if employee_insights:
# #                 response = "Diabetes risk insights for employees:\n"
# #                 for emp_id, prob, risk_status, future_status in employee_insights:
# #                     response += f"Employee ID {emp_id}: Predicted probability - {prob:.2f}, Status - {risk_status}, {future_status}\n"
# #                 response += f"\nHigh Risk Count: {high_risk_count}\nModerate Risk Count: {moderate_risk_count}\nLow Risk Count: {low_risk_count}"
# #             else:
# #                 response = "No employee data found."

# #             return response
# #         else:
# #             return "Employee ID information is not available in the data."
# #     except Exception as e:
# #         return f"Error: {str(e)}"

# # # Function to extract checkup ID from user message
# # def extract_checkup_id(user_msg):
# #     words = user_msg.split()
# #     for word in words:
# #         if word.isdigit():
# #             return int(word)
# #     return None
# # # Endpoint to handle user messages
# # @app.route('/send', methods=['POST'])
# # def handle_message():
# #     user_msg = request.form['msg'].lower()  # Convert to lowercase for consistency
# #     response = ""

# #     # Use NLP to extract tokens from the user's message
# #     doc = nlp(user_msg)
# #     tokens = [token.text for token in doc]

# #     # Check if any keyword in the user's message matches a keyword in the database
# #     for token in tokens:
# #         cursor.execute("SELECT function_name FROM keyword_functions WHERE keyword LIKE %s", (f"%{token}%",))
# #         result = cursor.fetchone()
        
# #         # Ensure all results are fetched before executing the next query
# #         cursor.fetchall()
        
# #         if result:
# #             function_name = result[0]
# #             if function_name in globals():
# #                 response = globals()[function_name](user_msg)
# #             else:
# #                 response = "Function not implemented."
# #             break
# #     else:
# #         response = "Keyword not recognized. Please try another query."

# #     return jsonify({'response': response})

# # if __name__ == '__main__':
# #     app.run(debug=True)
# from flask import Flask, request, jsonify
# import mysql.connector
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# import nltk
# import re

# # Initialize Flask app
# app = Flask(__name__)

# # Connect to MySQL database
# db = mysql.connector.connect(
#     host="localhost",
#     user="your_username",
#     password="your_password",
#     database="your_database"
# )

# cursor = db.cursor()

# # Load dataset and train model
# def load_data():
#     cursor.execute("SELECT * FROM chatbot_data")
#     data = cursor.fetchall()
#     texts = [row[1] for row in data]
#     labels = [row[2] for row in data]
#     return texts, labels

# texts, labels = load_data()

# # Preprocess text
# nltk.download('stopwords')
# stop_words = set(nltk.corpus.stopwords.words('english'))

# def preprocess_text(text):
#     text = re.sub(r'\W', ' ', text)
#     text = text.lower()
#     text = text.split()
#     text = [word for word in text if word not in stop_words]
#     text = ' '.join(text)
#     return text

# texts = [preprocess_text(text) for text in texts]

# # Train model
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(texts)
# model = MultinomialNB()
# model.fit(X, labels)

# # Chatbot route
# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('message')
#     processed_input = preprocess_text(user_input)
#     vectorized_input = vectorizer.transform([processed_input])
#     predicted_label = model.predict(vectorized_input)[0]

#     cursor.execute("SELECT response FROM chatbot_data WHERE intent=%s", (predicted_label,))
#     response = cursor.fetchone()[0]
    
#     # Log conversation
#     cursor.execute("INSERT INTO chatbot_logs (user_input, bot_response, intent) VALUES (%s, %s, %s)", (user_input, response, predicted_label))
#     db.commit()

#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)

























# from flask import Flask, request, jsonify, render_template
# import mysql.connector
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# import spacy
# from spellchecker import SpellChecker

# app = Flask(__name__)

# # MySQL connection configuration
# mysql_config = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '',
#     'database': 'demo_db',
#     'auth_plugin': 'mysql_native_password'
# }

# # Connect to MySQL
# db_connection = mysql.connector.connect(**mysql_config)
# cursor = db_connection.cursor()

# # Load English NLP model from spaCy
# nlp = spacy.load("en_core_web_sm")

# # Homepage route to serve HTML form
# @app.route('/')
# def index():
#     return render_template('index.html', chat_history=[])

# # Function to handle checkup queries
# def handle_checkup(user_msg):
#     checkup_id = extract_checkup_id(user_msg)
#     if checkup_id:
#         cursor.execute("SELECT bmi FROM checkup_form WHERE checkup_id = %s", (checkup_id,))
#         result = cursor.fetchone()
#         if result:
#             bmi = result[0]
#             return f"Your height (BMI) for checkup ID {checkup_id} is {bmi}."
#         else:
#             return f"No height information found for checkup ID {checkup_id}."
#     else:
#         return "Could not extract checkup ID from your query."

# # Function to count diabetes patients
# def count_diabetes_patients(user_msg):
#     cursor.execute("SELECT COUNT(*) FROM diabetes_table WHERE outcome = '1'")
#     result = cursor.fetchone()
#     if result:
#         diabetes_count = result[0]
#         return f"There are {diabetes_count} patients with diabetes in the training data."
#     else:
#         return "No diabetes patient information found in the training data."

# def predict_diabetes(user_msg):
#     try:
#         # Fetch data from the database
#         cursor.execute("SELECT * FROM diabetes_table")
#         data = cursor.fetchall()
#         column_names = [desc[0] for desc in cursor.description]
#         df = pd.DataFrame(data, columns=column_names)

#         # Ensure consistent feature set
#         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']

#         # Preparing the data for training
#         X = df[feature_columns]
#         y = df['outcome']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)

#         # Train the model
#         model = LogisticRegression()
#         model.fit(X_train_scaled, y_train)

#         # Example new data for prediction (replace with actual new data)
#         new_data_dict = {
#             'pregnancies': [5], 
#             'glucose': [166], 
#             'blood_pressure': [72], 
#             'skin_thickness': [19], 
#             'insulin': [175], 
#             'bmi': [25.8], 
#             'diabetes_pedigree_function': [0.587], 
#             'age': [51]
#         }
#         new_data_df = pd.DataFrame(new_data_dict)
#         new_data_scaled = scaler.transform(new_data_df)
#         prediction = model.predict_proba(new_data_scaled)

#         predicted_prob = prediction[0][1]
#         return f"The predicted probability of developing diabetes is {predicted_prob:.2f}."
#     except ValueError as e:
#         return f"Error: {str(e)}"

# def predict_future_diabetes(user_msg):
#     try:
#         # Fetch employee data from the database
#         cursor.execute("SELECT * FROM diabetes_table")
#         data = cursor.fetchall()
#         column_names = [desc[0] for desc in cursor.description]
#         df = pd.DataFrame(data, columns=column_names)

#         # Ensure consistent feature set
#         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']

#         # Preparing the data for training
#         X = df[feature_columns]
#         y = df['outcome']
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         # Train the model
#         model = LogisticRegression()
#         model.fit(X_scaled, y)

#         # Predict for all employees
#         if 'emp_id' in df.columns:
#             employee_insights = []
#             high_risk_count = 0
#             moderate_risk_count = 0
#             low_risk_count = 0

#             for emp_id in df['emp_id'].unique():
#                 emp_data = df[df['emp_id'] == emp_id][feature_columns]
#                 emp_data_scaled = scaler.transform(emp_data)
#                 prediction = model.predict_proba(emp_data_scaled)
#                 predicted_prob = prediction[:, 1].mean()  # Average probability across all data

#                 # Classify based on predicted probability
#                 if predicted_prob > 0.5:
#                     risk_status = "High risk"
#                     future_diabetes_patients = "Future diabetes patients"
#                     high_risk_count += 1
#                 elif predicted_prob > 0.2:  # Adjust thresholds as needed
#                     risk_status = "Moderate risk"
#                     future_diabetes_patients = "Future diabetes patients"
#                     moderate_risk_count += 1
#                 else:
#                     risk_status = "Low risk"
#                     future_diabetes_patients = "Not future diabetes patients"
#                     low_risk_count += 1

#                 employee_insights.append((emp_id, predicted_prob, risk_status, future_diabetes_patients))

#             # Format response
#             if employee_insights:
#                 response = "Diabetes risk insights for employees:\n"
#                 for emp_id, prob, risk_status, future_status in employee_insights:
#                     response += f"Employee ID {emp_id}: Predicted probability - {prob:.2f}, Status - {risk_status}, {future_status}\n"
#                 response += f"\nHigh Risk Count: {high_risk_count}\nModerate Risk Count: {moderate_risk_count}\nLow Risk Count: {low_risk_count}"
#             else:
#                 response = "No employee data found."

#             return response
#         else:
#             return "Employee ID information is not available in the data."
#     except Exception as e:
#         return f"Error: {str(e)}"

# # Function to extract checkup ID from user message
# def extract_checkup_id(user_msg):
#     words = user_msg.split()
#     for word in words:
#         if word.isdigit():
#             return int(word)
#     return None
# # Endpoint to handle user messages
# @app.route('/send', methods=['POST'])
# def handle_message():
#     user_msg = request.form['msg'].lower()  # Convert to lowercase for consistency
#     response = ""

#     # Use NLP to extract tokens from the user's message
#     doc = nlp(user_msg)
#     tokens = [token.text for token in doc]

#     # Check if any keyword in the user's message matches a keyword in the database
#     for token in tokens:
#         cursor.execute("SELECT function_name FROM keyword_functions WHERE keyword LIKE %s", (f"%{token}%",))
#         result = cursor.fetchone()
        
#         # Ensure all results are fetched before executing the next query
#         cursor.fetchall()
        
#         if result:
#             function_name = result[0]
#             if function_name in globals():
#                 response = globals()[function_name](user_msg)
#             else:
#                 response = "Function not implemented."
#             break
#     else:
#         response = "Keyword not recognized. Please try another query."

#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)












# from flask import Flask, request, jsonify, render_template
# import mysql.connector
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# import spacy

# app = Flask(__name__)

# # MySQL connection configuration
# mysql_config = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '',
#     'database': 'demo_db',
#     'auth_plugin': 'mysql_native_password'
# }

# # Connect to MySQL
# db_connection = mysql.connector.connect(**mysql_config)
# cursor = db_connection.cursor()

# # Load English NLP model from spaCy
# nlp = spacy.load("en_core_web_sm")

# # Homepage route to serve HTML form
# @app.route('/')
# def index():
#     cursor.execute("SELECT complaint FROM complaints")
#     complaints = [row[0] for row in cursor.fetchall()]
#     return render_template('index.html', chat_history=[], complaints=complaints)

# # Function to handle checkup queries
# def handle_checkup(user_msg):
#     checkup_id = extract_checkup_id(user_msg)
#     if checkup_id:
#         cursor.execute("SELECT bmi FROM checkup_form WHERE checkup_id = %s", (checkup_id,))
#         result = cursor.fetchone()
#         if result:
#             bmi = result[0]
#             return f"Your height (BMI) for checkup ID {checkup_id} is {bmi}."
#         else:
#             return f"No height information found for checkup ID {checkup_id}."
#     else:
#         return "Could not extract checkup ID from your query."

# # Function to count diabetes patients
# def count_diabetes_patients(user_msg):
#     cursor.execute("SELECT COUNT(*) FROM diabetes_table WHERE outcome = '1'")
#     result = cursor.fetchone()
#     if result:
#         diabetes_count = result[0]
#         return f"There are {diabetes_count} patients with diabetes in the training data."
#     else:
#         return "No diabetes patient information found in the training data."

# # Function to predict diabetes
# def predict_diabetes(user_msg):
#     try:
#         cursor.execute("SELECT * FROM diabetes_table")
#         data = cursor.fetchall()
#         column_names = [desc[0] for desc in cursor.description]
#         df = pd.DataFrame(data, columns=column_names)

#         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']
#         X = df[feature_columns]
#         y = df['outcome']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)

#         model = LogisticRegression()
#         model.fit(X_train_scaled, y_train)

#         new_data_dict = {
#             'pregnancies': [5], 
#             'glucose': [166], 
#             'blood_pressure': [72], 
#             'skin_thickness': [19], 
#             'insulin': [175], 
#             'bmi': [25.8], 
#             'diabetes_pedigree_function': [0.587], 
#             'age': [51]
#         }
#         new_data_df = pd.DataFrame(new_data_dict)
#         new_data_scaled = scaler.transform(new_data_df)
#         prediction = model.predict_proba(new_data_scaled)

#         predicted_prob = prediction[0][1]
#         return f"The predicted probability of developing diabetes is {predicted_prob:.2f}."
#     except ValueError as e:
#         return f"Error: {str(e)}"

# # Function to predict future diabetes risk for employees
# def predict_future_diabetes(user_msg):
#     try:
#         cursor.execute("SELECT * FROM diabetes_table")
#         data = cursor.fetchall()
#         column_names = [desc[0] for desc in cursor.description]
#         df = pd.DataFrame(data, columns=column_names)

#         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']
#         X = df[feature_columns]
#         y = df['outcome']
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         model = LogisticRegression()
#         model.fit(X_scaled, y)

#         if 'emp_id' in df.columns:
#             employee_insights = []
#             high_risk_count = 0
#             moderate_risk_count = 0
#             low_risk_count = 0

#             for emp_id in df['emp_id'].unique():
#                 emp_data = df[df['emp_id'] == emp_id][feature_columns]
#                 emp_data_scaled = scaler.transform(emp_data)
#                 prediction = model.predict_proba(emp_data_scaled)
#                 predicted_prob = prediction[:, 1].mean()

#                 if predicted_prob > 0.5:
#                     risk_status = "High risk"
#                     future_diabetes_patients = "Future diabetes patients"
#                     high_risk_count += 1
#                 elif predicted_prob > 0.2:
#                     risk_status = "Moderate risk"
#                     future_diabetes_patients = "Future diabetes patients"
#                     moderate_risk_count += 1
#                 else:
#                     risk_status = "Low risk"
#                     future_diabetes_patients = "Not future diabetes patients"
#                     low_risk_count += 1

#                 employee_insights.append((emp_id, predicted_prob, risk_status, future_diabetes_patients))

#             if employee_insights:
#                 response = "Diabetes risk insights for employees:\n"
#                 for emp_id, prob, risk_status, future_status in employee_insights:
#                     response += f"Employee ID {emp_id}: Predicted probability - {prob:.2f}, Status - {risk_status}, {future_status}\n"
#                 response += f"\nHigh Risk Count: {high_risk_count}\nModerate Risk Count: {moderate_risk_count}\nLow Risk Count: {low_risk_count}"
#             else:
#                 response = "No employee data found."
#             return response
#         else:
#             return "Employee ID information is not available in the data."
#     except Exception as e:
#         return f"Error: {str(e)}"

# # Function to extract checkup ID from user message
# def extract_checkup_id(user_msg):
#     words = user_msg.split()
#     for word in words:
#         if word.isdigit():
#             return int(word)
#     return None

# # Endpoint to handle user messages
# @app.route('/send', methods=['POST'])
# def handle_message():
#     user_msg = request.form['msg'].lower()
#     response = ""

#     doc = nlp(user_msg)
#     tokens = [token.text for token in doc]

#     for token in tokens:
#         cursor.execute("SELECT function_name FROM keyword_functions WHERE keyword LIKE %s", (f"%{token}%",))
#         result = cursor.fetchone()
        
#         cursor.fetchall()
        
#         if result:
#             function_name = result[0]
#             if function_name in globals():
#                 response = globals()[function_name](user_msg)
#             else:
#                 response = "Function not implemented."
#             break
#     else:
#         response = "Keyword not recognized. Please try another query."

#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# import mysql.connector
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# import spacy
# import logging

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # MySQL connection configuration
# mysql_config = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '',
#     'database': 'demo_db',
#     'auth_plugin': 'mysql_native_password'
# }

# # Load English NLP model from spaCy
# nlp = spacy.load("en_core_web_sm")

# # Homepage route to serve HTML form
# @app.route('/')
# def index():
#     connection = mysql.connector.connect(**mysql_config)
#     cursor = connection.cursor()
#     cursor.execute("SELECT complaint FROM complaints")
#     complaints = [row[0] for row in cursor.fetchall()]
#     connection.close()
#     return render_template('index.html', chat_history=[], complaints=complaints)

# # Function to predict diagnosis based on symptoms
# # Function to predict diagnosis based on symptoms
# def predict_diagnosis(symptoms):
#     try:
#         connection = mysql.connector.connect(**mysql_config)
#         cursor = connection.cursor()
#         cursor.execute("SELECT * FROM medical_data_table")
#         data = cursor.fetchall()
#         column_names = [desc[0] for desc in cursor.description]
#         df = pd.DataFrame(data, columns=column_names)
#         connection.close()

#         feature_columns = ['fever', 'headache', 'fatigue', 'nausea', 'chills', 'muscle_pain', 'sore_throat']
#         X = df[feature_columns]
#         y = df['diagnosis']

#         # Train the model
#         model = DecisionTreeClassifier(random_state=42)
#         model.fit(X, y)

#         new_data = pd.DataFrame([symptoms], columns=feature_columns)
#         prediction = model.predict_proba(new_data)

#         # Get the highest probability diagnosis
#         top_prediction_index = prediction[0].argmax()
#         top_diagnosis = model.classes_[top_prediction_index]
#         top_probability = prediction[0][top_prediction_index]

#         response = f"The predicted diagnosis based on your symptoms is {top_diagnosis}: {top_probability:.2f}%"
#         return response
#     except Exception as e:
#         return f"Error: {str(e)}"

# # Endpoint to handle user messages
# @app.route('/send', methods=['POST'])
# def handle_message():
#     user_msg = request.form.get('msg', '').lower()
#     selected_symptoms = request.form.getlist('selected_symptoms[]')

#     if selected_symptoms:
#         # Convert selected symptoms to the required format for prediction
#         symptoms = {symptom: 1 if symptom in selected_symptoms else 0 for symptom in ['fever', 'headache', 'fatigue', 'nausea', 'chills', 'muscle_pain', 'sore_throat']}
#         response = predict_diagnosis(symptoms)
#     else:
#         response = ""

#         # Check if user mentions fever
#         if 'fever' in user_msg:
#             # Fetch symptoms dynamically from the columns of the medical_data_table
#             connection = mysql.connector.connect(**mysql_config)
#             cursor = connection.cursor()
#             cursor.execute("SHOW COLUMNS FROM medical_data_table")
#             symptom_columns = [row[0] for row in cursor.fetchall() if row[0] in ['fever', 'headache', 'fatigue', 'nausea', 'chills', 'muscle_pain', 'sore_throat']]
#             connection.close()

#             response = (
#                 "I see you mentioned a fever. Please select the symptoms you are experiencing from the list below:\n" +
#                 "\n".join(f"<div class='message-container complaint-message' data-symptom='{symptom}'>{symptom}</div>" for symptom in symptom_columns) +
#                 "\n\nYou can respond with the numbers or the names of the symptoms. For example, you can say: 'I have headache and fatigue.'"
#             )
#         else:
#             # Process the message for diagnosis prediction
#             response = predict_diagnosis(user_msg)

#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)
























# from flask import Flask, request, jsonify, render_template
# import mysql.connector
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# import spacy
# import logging

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # MySQL connection configuration
# mysql_config = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '',
#     'database': 'demo_db',
#     'auth_plugin': 'mysql_native_password'
# }

# # Load English NLP model from spaCy
# nlp = spacy.load("en_core_web_sm")

# # Homepage route to serve HTML form
# @app.route('/')
# def index():
#     connection = mysql.connector.connect(**mysql_config)
#     cursor = connection.cursor()
#     cursor.execute("SELECT complaint FROM complaints")
#     complaints = [row[0] for row in cursor.fetchall()]
#     connection.close()
#     return render_template('index.html', chat_history=[], complaints=complaints)

# # Function to predict diagnosis based on symptoms
# def handle_checkup(user_msg):
#     checkup_id = extract_checkup_id(user_msg)
#     if checkup_id:
#         connection = mysql.connector.connect(**mysql_config)
#         cursor = connection.cursor()
#         cursor.execute("SELECT bmi FROM checkup_form WHERE checkup_id = %s", (checkup_id,))
#         result = cursor.fetchone()
#         connection.close()
#         if result:
#             bmi = result[0]
#             return f"Your height (BMI) for checkup ID {checkup_id} is {bmi}."
#         else:
#             return f"No height information found for checkup ID {checkup_id}."
#     else:
#         return "Could not extract checkup ID from your query."

# # Function to count diabetes patients
# def count_diabetes_patients(user_msg):
#     connection = mysql.connector.connect(**mysql_config)
#     cursor = connection.cursor()
#     cursor.execute("SELECT COUNT(*) FROM diabetes_table WHERE outcome = '1'")
#     result = cursor.fetchone()
#     connection.close()
#     if result:
#         diabetes_count = result[0]
#         return f"There are {diabetes_count} patients with diabetes in the training data."
#     else:
#         return "No diabetes patient information found in the training data."

# # Function to predict diabetes
# def predict_diabetes(user_msg):
#     try:
#         connection = mysql.connector.connect(**mysql_config)
#         cursor = connection.cursor()
#         cursor.execute("SELECT * FROM diabetes_table")
#         data = cursor.fetchall()
#         column_names = [desc[0] for desc in cursor.description]
#         df = pd.DataFrame(data, columns=column_names)
#         connection.close()

#         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']
#         X = df[feature_columns]
#         y = df['outcome']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)

#         model = LogisticRegression()
#         model.fit(X_train_scaled, y_train)

#         new_data_dict = {
#             'pregnancies': [5], 
#             'glucose': [166], 
#             'blood_pressure': [72], 
#             'skin_thickness': [19], 
#             'insulin': [175], 
#             'bmi': [25.8], 
#             'diabetes_pedigree_function': [0.587], 
#             'age': [51]
#         }
#         new_data_df = pd.DataFrame(new_data_dict)
#         new_data_scaled = scaler.transform(new_data_df)
#         prediction = model.predict_proba(new_data_scaled)

#         predicted_prob = prediction[0][1]
#         return f"The predicted probability of developing diabetes is {predicted_prob:.2f}."
#     except ValueError as e:
#         return f"Error: {str(e)}"

# # Function to predict future diabetes risk for employees
# def predict_future_diabetes(user_msg):
#     try:
#         connection = mysql.connector.connect(**mysql_config)
#         cursor = connection.cursor()
#         cursor.execute("SELECT * FROM diabetes_table")
#         data = cursor.fetchall()
#         column_names = [desc[0] for desc in cursor.description]
#         df = pd.DataFrame(data, columns=column_names)
#         connection.close()

#         feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']
#         X = df[feature_columns]
#         y = df['outcome']
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         model = LogisticRegression()
#         model.fit(X_scaled, y)

#         if 'emp_id' in df.columns:
#             employee_insights = []
#             high_risk_count = 0
#             moderate_risk_count = 0
#             low_risk_count = 0

#             for emp_id in df['emp_id'].unique():
#                 emp_data = df[df['emp_id'] == emp_id][feature_columns]
#                 emp_data_scaled = scaler.transform(emp_data)
#                 prediction = model.predict_proba(emp_data_scaled)
#                 predicted_prob = prediction[:, 1].mean()

#                 if predicted_prob > 0.5:
#                     risk_status = "High risk"
#                     future_diabetes_patients = "Future diabetes patients"
#                     high_risk_count += 1
#                 elif predicted_prob > 0.2:
#                     risk_status = "Moderate risk"
#                     future_diabetes_patients = "Future diabetes patients"
#                     moderate_risk_count += 1
#                 else:
#                     risk_status = "Low risk"
#                     future_diabetes_patients = "Not future diabetes patients"
#                     low_risk_count += 1

#                 employee_insights.append((emp_id, predicted_prob, risk_status, future_diabetes_patients))

#             if employee_insights:
#                 response = "Diabetes risk insights for employees:\n"
#                 for emp_id, prob, risk_status, future_status in employee_insights:
#                     response += f"Employee ID {emp_id}: Predicted probability - {prob:.2f}, Status - {risk_status}, {future_status}\n"
#                 response += f"\nHigh Risk Count: {high_risk_count}\nModerate Risk Count: {moderate_risk_count}\nLow Risk Count: {low_risk_count}"
#             else:
#                 response = "No employee data found."
#             return response
#         else:
#             return "Employee ID information is not available in the data."
#     except Exception as e:
#         return f"Error: {str(e)}"

# # Function to extract checkup ID from user message
# def extract_checkup_id(user_msg):
#     words = user_msg.split()
#     for word in words:
#         if word.isdigit():
#             return int(word)
#     return None

# def predict_diagnosis(symptoms):
#     try:
#         connection = mysql.connector.connect(**mysql_config)
#         cursor = connection.cursor()
#         cursor.execute("SELECT * FROM medical_data_table")
#         data = cursor.fetchall()
#         column_names = [desc[0] for desc in cursor.description]
#         df = pd.DataFrame(data, columns=column_names)
#         connection.close()

#         logging.debug(f"Fetched Data from DB:\n{df}")

#         feature_columns = ['fever', 'headache', 'fatigue', 'nausea', 'chills', 'muscle_pain', 'sore_throat']
#         X = df[feature_columns]
#         y = df['diagnosis']

#         model = DecisionTreeClassifier(random_state=42)
#         model.fit(X, y)

#         new_data = pd.DataFrame([symptoms], columns=feature_columns)
#         logging.debug(f"New Data for Prediction:\n{new_data}")

#         prediction = model.predict_proba(new_data)
#         logging.debug(f"Prediction Probabilities:\n{prediction}")

#         top_prediction_index = prediction[0].argmax()
#         top_diagnosis = model.classes_[top_prediction_index]
#         top_probability = prediction[0][top_prediction_index]

#         response = f"The predicted diagnosis based on your symptoms is {top_diagnosis}: {top_probability:.2f}%"
#         return response
#     except Exception as e:
#         logging.error(f"Error in predict_diagnosis: {str(e)}")
#         return f"Error: {str(e)}"

# # Endpoint to handle user messages
# @app.route('/send', methods=['POST'])
# def handle_message():
#     user_msg = request.form.get('msg', '').lower()
#     selected_symptoms = request.form.getlist('selected_symptoms[]')

#     response = ""
#     if selected_symptoms:
#         # Convert selected symptoms to the required format for prediction
#         symptoms = {symptom: 1 if symptom in selected_symptoms else 0 for symptom in ['fever', 'headache', 'fatigue', 'nausea', 'chills', 'muscle_pain', 'sore_throat']}
#         response = predict_diagnosis(symptoms)
#     else:
#         connection = mysql.connector.connect(**mysql_config)
#         cursor = connection.cursor()

#         doc = nlp(user_msg)
#         tokens = [token.text for token in doc]

#         for token in tokens:
#             cursor.execute("SELECT function_name FROM keyword_functions WHERE keyword LIKE %s", (f"%{token}%",))
#             result = cursor.fetchone()
#             if result:
#                 function_name = result[0]
#                 if function_name in globals():
#                     response = globals()[function_name](user_msg)
#                 else:
#                     response = "Function not implemented."
#                 break
#         else:
#             if 'fever' in user_msg:
#                 # Fetch symptoms dynamically from the columns of the medical_data_table
#                 cursor.execute("SHOW COLUMNS FROM medical_data_table")
#                 symptom_columns = [row[0] for row in cursor.fetchall() if row[0] in ['fever', 'headache', 'fatigue', 'nausea', 'chills', 'muscle_pain', 'sore_throat']]

#                 response = (
#                     "I see you mentioned a fever. Please select the symptoms you are experiencing from the list below:\n" +
#                     "\n".join(f"<div class='message-container complaint-message' data-symptom='{symptom}'>{symptom}</div>" for symptom in symptom_columns) +
#                     "\n\nYou can respond with the numbers or the names of the symptoms. For example, you can say: 'I have headache and fatigue.'"
#                 )
#             else:
#                 response = "I couldn't find a relevant function to process your request."

#         connection.close()

#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# import mysql.connector
# import spacy
# import logging

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # MySQL connection configuration
# mysql_config = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '',
#     'database': 'demo_db',
#     'auth_plugin': 'mysql_native_password'
# }

# # Load English NLP model from spaCy
# nlp = spacy.load("en_core_web_sm")

# # Homepage route to serve HTML form
# @app.route('/')
# def index():
#     return render_template('index.html', chat_history=[], complaints=[])

# # Function to handle fever-related suggestions
# def handle_fever():
#     connection = mysql.connector.connect(**mysql_config)
#     cursor = connection.cursor()
#     cursor.execute("SELECT complaint FROM complaints WHERE complaint LIKE '%FEVER%'")
#     symptoms = [row[0] for row in cursor.fetchall()]
#     connection.close()
#     return (
#         "I see you mentioned a fever. Please select the symptoms you are experiencing from the list below:\n" +
#         "\n".join(f"<div class='message-container complaint-message' data-symptom='{symptom}'>{symptom}</div>" for symptom in symptoms) +
#         "\n\nYou can respond with the numbers or the names of the symptoms. For example, you can say: 'I have headache and fatigue.'"
#     )

# # Endpoint to handle user messages
# @app.route('/send', methods=['POST'])
# def handle_message():
#     user_msg = request.form.get('msg', '').lower()
#     selected_complaint_id = request.form.get('complaint_id')
#     selected_symptoms = request.form.getlist('selected_symptoms[]')

#     response = ""
#     connection = mysql.connector.connect(**mysql_config)
#     cursor = connection.cursor()

#     if selected_symptoms:
#         # Check if selected symptoms match any complaint
#         cursor.execute("""
#             SELECT complaint_desc 
#             FROM compl 
#             WHERE FIND_IN_SET(%s, complaint)
#         """, (', '.join(selected_symptoms),))
#         result = cursor.fetchone()
#         connection.close()
#         if result:
#             response = f"Based on the symptoms you selected ({', '.join(selected_symptoms)}), here is the related information: {result[0]}"
#         else:
#             response = "No matching complaint description found for the selected symptoms."
#     elif selected_complaint_id:
#         cursor.execute("SELECT complaint_desc FROM compl WHERE complaint_id = %s", (selected_complaint_id,))
#         result = cursor.fetchone()
#         connection.close()
#         if result:
#             response = result[0]
#         else:
#             response = "No description found for the selected complaint."
#     else:
#         doc = nlp(user_msg)
#         tokens = [token.text for token in doc]

#         for token in tokens:
#             cursor.execute("SELECT function_name FROM keyword_functions WHERE keyword LIKE %s", (f"%{token}%",))
#             result = cursor.fetchone()
#             if result:
#                 function_name = result[0]
#                 if function_name in globals():
#                     response = globals()[function_name](user_msg)
#                 else:
#                     response = "Function not implemented."
#                 break
#         else:
#             if 'fever' in user_msg:
#                 response = handle_fever()
#             else:
#                 response = "I couldn't find a relevant function to process your request."

#         connection.close()

#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# import mysql.connector
# import spacy
# import logging

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # MySQL connection configuration
# mysql_config = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '',
#     'database': 'demo_db',
#     'auth_plugin': 'mysql_native_password'
# }

# # Load English NLP model from spaCy
# nlp = spacy.load("en_core_web_sm")

# # Homepage route to serve HTML form
# @app.route('/')
# def index():
#     return render_template('index.html', chat_history=[], complaints=[])

# # Function to handle fever-related suggestions
# def handle_fever():
#     connection = mysql.connector.connect(**mysql_config)
#     cursor = connection.cursor()
#     cursor.execute("SELECT complaint FROM complaints WHERE complaint LIKE '%FEVER%'")
#     symptoms = [row[0] for row in cursor.fetchall()]
#     connection.close()
#     return (
#         "I see you mentioned a fever. Please select the symptoms you are experiencing from the list below:\n" +
#         "\n".join(f"<div class='message-container complaint-message' data-symptom='{symptom}'>{symptom}</div>" for symptom in symptoms) +
#         "\n\nYou can respond with the numbers or the names of the symptoms. For example, you can say: 'I have headache and fatigue.'"
#     )

# # Endpoint to handle user messages
# @app.route('/send', methods=['POST'])
# def handle_message():
#     user_msg = request.form.get('msg', '').lower()
#     selected_complaint_id = request.form.get('complaint_id')
#     selected_symptoms = request.form.getlist('selected_symptoms[]')

#     response = ""
#     connection = mysql.connector.connect(**mysql_config)
#     cursor = connection.cursor()

#     if selected_symptoms:
#         # Check if selected symptoms match any complaint
#         cursor.execute("""
#             SELECT complaint_desc 
#             FROM compl
#             WHERE FIND_IN_SET(%s, complaint)
#         """, (', '.join(selected_symptoms),))
#         result = cursor.fetchone()
#         connection.close()
#         if result:
#             response = f"Based on the symptoms you selected ({', '.join(selected_symptoms)}), here is the related information: {result[0]}"
#         else:
#             response = "No matching complaint description found for the selected symptoms."
#     elif selected_complaint_id:
#         cursor.execute("SELECT complaint_desc FROM compl WHERE complaint_id = %s", (selected_complaint_id,))
#         result = cursor.fetchone()
#         connection.close()
#         if result:
#             response = result[0]
#         else:
#             response = "No description found for the selected complaint."
#     else:
#         doc = nlp(user_msg)
#         tokens = [token.text for token in doc]

#         function_executed = False
#         for token in tokens:
#             cursor.execute("SELECT function_name FROM keyword_functions WHERE keyword LIKE %s", (f"%{token}%",))
#             result = cursor.fetchone()
#             if result:
#                 function_name = result[0]
#                 if function_name in globals():
#                     # Call the function without any arguments
#                     response = globals()[function_name]()
#                     function_executed = True
#                 else:
#                     response = "Function not implemented."
#                 break

#         if not function_executed:
#             response = "I couldn't find a relevant function to process your request."

#         connection.close()

#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, request, jsonify, render_template
# import mysql.connector
# import spacy
# import logging

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # MySQL connection configuration
# mysql_config = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '',
#     'database': 'demo_db',
#     'auth_plugin': 'mysql_native_password'
# }

# # Load English NLP model from spaCy
# nlp = spacy.load("en_core_web_sm")

# # Homepage route to serve HTML form
# @app.route('/')
# def index():
#     return render_template('index.html', chat_history=[], complaints=[])

# # Function to handle suggestions based on keyword and key
# def handle_keyword(keyword, key):
#     logging.debug(f"Handling keyword: {keyword}, key: {key}")
#     connection = mysql.connector.connect(**mysql_config)
#     cursor = connection.cursor()
#     cursor.execute("SELECT complaint FROM complaints WHERE complaint LIKE %s", (f'%{key}%',))
#     symptoms = [row[0] for row in cursor.fetchall()]
#     connection.close()
#     logging.debug(f"Symptoms found: {symptoms}")
#     if symptoms:
#         return (
#             f"I see you mentioned {keyword}. Please select the symptoms you are experiencing from the list below:\n" +
#             "\n".join(f"<div class='message-container complaint-message' data-symptom='{symptom}'>{symptom}</div>" for symptom in symptoms) +
#             "\n\nYou can respond with the numbers or the names of the symptoms. For example, you can say: 'I have headache and fatigue.'"
#         )
#     else:
#         return f"No symptoms found for the keyword {keyword}."

# # Endpoint to handle user messages
# @app.route('/send', methods=['POST'])
# def handle_message():
#     user_msg = request.form.get('msg', '').lower()
#     response = ""

#     logging.debug(f"Received user message: {user_msg}")

#     connection = mysql.connector.connect(**mysql_config)
#     cursor = connection.cursor()

#     doc = nlp(user_msg)
#     tokens = [token.text for token in doc]
#     logging.debug(f"Tokens extracted from user message: {tokens}")
#     keyword_found = False
#     cursor.execute("SELECT keyword, `key` FROM keyword_functions")
#     all_keywords = cursor.fetchall()
    
#     logging.debug(f"Fetched keywords from database: {all_keywords}")
    
#     for token in tokens:
#         logging.debug(f"Processing token: {token}")
#         for row in all_keywords:
#             keyword_list = row[0].split(',')
#             logging.debug(f"Checking against keyword list: {keyword_list}")
#             if token in keyword_list:
#                 keyword = token
#                 key = row[1]
#                 logging.debug(f"Keyword matched: {keyword}, key: {key}")
#                 response = handle_keyword(keyword, key)
#                 keyword_found = True
#                 break
#         if keyword_found:
#             break
    
#     if not keyword_found:
#         response = "I couldn't find a relevant keyword to process your request."

#     connection.close()
#     logging.debug(f"Response generated: {response}")
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# import mysql.connector
# import spacy
# import logging
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # MySQL connection configuration
# mysql_config = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '',
#     'database': 'mh',
#     'auth_plugin': 'mysql_native_password'
# }

# # Load English NLP model from spaCy
# nlp = spacy.load("en_core_web_sm")

# # Function to load data from MySQL
# def load_data():
#     connection = mysql.connector.connect(**mysql_config)
#     query = "SELECT * FROM medical_data_table"
#     df = pd.read_sql(query, connection)
#     connection.close()
#     return df

# # Function to handle suggestions based on keyword and key
# def handle_keyword(keyword, key):
#     logging.debug(f"Handling keyword: {keyword}, key: {key}")
#     connection = mysql.connector.connect(**mysql_config)
#     cursor = connection.cursor()
#     cursor.execute("SELECT complaint FROM complaints WHERE complaint LIKE %s", (f'%{key}%',))
#     symptoms = [row[0] for row in cursor.fetchall()]
#     connection.close()
#     logging.debug(f"Symptoms found: {symptoms}")
#     if symptoms:
#         return (
#             f"I see you mentioned {keyword}. Please select the symptoms you are experiencing from the list below:\n" +
#             "\n".join(f"<div class='message-container complaint-message' data-symptom='{symptom}'>{symptom}</div>" for symptom in symptoms) +
#             "\n\nYou can respond with the numbers or the names of the symptoms. For example, you can say: 'I have headache and fatigue.'"
#         )
#     else:
#         return f"No symptoms found for the keyword {keyword}."

# # Function to predict diagnosis based on symptoms
# def predict_diagnosis(symptoms):
#     try:
#         connection = mysql.connector.connect(**mysql_config)
#         cursor = connection.cursor()
#         cursor.execute("SELECT * FROM medical_data_table")
#         data = cursor.fetchall()
#         column_names = [desc[0] for desc in cursor.description]
#         df = pd.DataFrame(data, columns=column_names)
#         connection.close()

#         logging.debug(f"Fetched Data from DB:\n{df}")

#         # Ensure all feature columns are numeric
#         feature_columns = ['fever', 'headache', 'fatigue', 'nausea', 'chills', 'muscle_pain', 'sore_throat', 'TOOTH PAIN', 'BODY PAIN']
#         for col in feature_columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
        
#         # Drop rows with NaN values
#         df = df.dropna()

#         X = df[feature_columns]
#         y = df['diagnosis']

#         model = DecisionTreeClassifier(random_state=42)
#         model.fit(X, y)

#         new_data = pd.DataFrame([symptoms], columns=feature_columns)
#         logging.debug(f"New Data for Prediction:\n{new_data}")

#         prediction = model.predict_proba(new_data)
#         logging.debug(f"Prediction Probabilities:\n{prediction}")

#         top_prediction_index = prediction[0].argmax()
#         top_diagnosis = model.classes_[top_prediction_index]
#         top_probability = prediction[0][top_prediction_index]

#         response = f"The predicted diagnosis based on your symptoms is {top_diagnosis}: {top_probability:.2f}%"
#         return response
#     except Exception as e:
#         logging.error(f"Error in predict_diagnosis: {str(e)}")
#         return f"Error: {str(e)}"

# # Homepage route to serve HTML form
# @app.route('/')
# def index():
#     return render_template('index.html', chat_history=[], complaints=[])

# # Endpoint to handle user messages
# @app.route('/send', methods=['POST'])
# def handle_message():
#     user_msg = request.form.get('msg', '').lower()
#     selected_symptoms = request.form.getlist('selected_symptoms[]')
#     response = ""

#     logging.debug(f"Received user message: {user_msg}")
#     logging.debug(f"Selected symptoms: {selected_symptoms}")

#     if selected_symptoms:
#         symptoms = {symptom: 1 for symptom in selected_symptoms}
#         response = predict_diagnosis(symptoms)
#     else:
#         connection = mysql.connector.connect(**mysql_config)
#         cursor = connection.cursor()

#         doc = nlp(user_msg)
#         tokens = [token.text for token in doc]
#         logging.debug(f"Tokens extracted from user message: {tokens}")
#         keyword_found = False
#         cursor.execute("SELECT  keyword, `key` FROM keyword_functions")
#         all_keywords = cursor.fetchall()
        
#         logging.debug(f"Fetched keywords from database: {all_keywords}")
        
#         for token in tokens:
#             logging.debug(f"Processing token: {token}")
#             for row in all_keywords:
#                 keyword_list = row[0].split(',')
#                 logging.debug(f"Checking against keyword list: {keyword_list}")
#                 if token in keyword_list:
#                     keyword = token
#                     key = row[1]
#                     logging.debug(f"Keyword matched: {keyword}, key: {key}")
#                     response = handle_keyword(keyword, key)
#                     keyword_found = True
#                     break
#             if keyword_found:
#                 break
        
#         if not keyword_found:
#             response = "I couldn't find a relevant keyword to process your request."

#         connection.close()

#     logging.debug(f"Response generated: {response}")
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)