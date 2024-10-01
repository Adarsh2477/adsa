from flask import Flask, request, jsonify, render_template
import mysql.connector
import spacy
import logging
import pandas as pd
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import difflib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# MySQL connection configuration
mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'mh',
    'auth_plugin': 'mysql_native_password'
}

# Load English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")

# Function to extract checkup ID from user message
def extract_checkup_id(user_msg):
    words = user_msg.split()
    for word in words:
        if word.isdigit():
            return int(word)
    return None

def load_data_from_excel():
    try:
        # Replace 'your_file.xlsx' with the path to your actual Excel file
        df = pd.read_excel('symptom_diagnosis_data.xlsx')
        return df
    except Exception as e:
        logging.error(f"Error loading data from Excel: {str(e)}")
        return pd.DataFrame()
    

# Function to load data from MySQL
def load_data():
    try:
        connection = mysql.connector.connect(**mysql_config)
        query = "SELECT * FROM medical_data_table"
        df = pd.read_sql(query, connection)
        connection.close()
        return df
    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
        return pd.DataFrame()

# Function to handle suggestions based on keyword and key
def handle_keyword(keyword, key):
    logging.debug(f"Handling keyword: {keyword}, key: {key}")
    try:
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()
        cursor.execute("SELECT complaint FROM complaints WHERE complaint LIKE %s", (f'%{key}%',))
        symptoms = [row[0] for row in cursor.fetchall()]
        connection.close()
        logging.debug(f"Symptoms found: {symptoms}")
        if symptoms:
            return (
                f"I see you mentioned {keyword}. Please select the symptoms you are experiencing from the list below:\n" +
                "\n".join(f"<div class='message-container complaint-message' data-symptom='{symptom}'>{symptom}</div>" for symptom in symptoms) +
                "\n\nYou can respond with the numbers or the names of the symptoms. For example, you can say: 'I have headache and fatigue.'"
            )
        else:
            return f"No symptoms found for the keyword {keyword}."
    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
        return "Error retrieving symptoms."
def handle_disease(keyword, key):
    logging.debug(f"Handling keyword: {keyword}, key: {key}")
    try:
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()
        cursor.execute("SELECT complaint FROM complaints WHERE complaint LIKE %s", (f'%{key}%',))
        symptoms = [row[0] for row in cursor.fetchall()]
        connection.close()
        logging.debug(f"Symptoms found: {symptoms}")
        if symptoms:
            return (
                f"I see you mentioned {keyword}. Please select the symptoms you are experiencing from the list below:\n" +
                "\n".join(f"<div class='message-container complaint-message' data-symptom='{symptom}'>{symptom}</div>" for symptom in symptoms) +
                "\n\nYou can respond with the numbers or the names of the symptoms. For example, you can say: 'I have headache and fatigue.'"
            )
        else:
            return f"No symptoms found for the keyword {keyword}."
    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
        return "Error retrieving symptoms."    
    
    
def predict_diagnosis(symptoms):
    try:
        df = load_data_from_excel()  # Load data from the Excel file
        if df.empty:
            return "Error loading data."

        # Dynamically get feature columns from the DataFrame
        feature_columns = [col for col in df.columns if col != 'diagnosis']

        # Check for missing columns in the DataFrame
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing columns in data: {missing_columns}")
            return f"Feature columns missing in data: {', '.join(missing_columns)}"

        # Convert selected columns to numeric
        df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        # Ensure 'diagnosis' column exists in DataFrame
        if 'diagnosis' not in df.columns:
            logging.error("Missing 'diagnosis' column in data")
            return "Error: 'diagnosis' column is missing in the data."

        # Convert symptoms list to dictionary with all feature columns
        symptoms_dict = {col: 0 for col in feature_columns}
        for symptom in symptoms:
            # Match symptom to the closest column name
            matched_column = difflib.get_close_matches(symptom, feature_columns, n=1, cutoff=0.8)
            if matched_column:
                symptoms_dict[matched_column[0]] = 1

        # Align columns between DataFrame and symptoms dictionary
        symptoms_series = pd.Series(symptoms_dict)
        df_aligned, symptoms_aligned = df[feature_columns].align(symptoms_series, axis=1, fill_value=0)

        # Filter the DataFrame to find rows that match the symptoms
        matching_row = df[df_aligned.eq(symptoms_aligned, axis=1).all(axis=1)]

        if not matching_row.empty:
            # Exact match found
            diagnosis = matching_row['diagnosis'].values[0]
            response = f"The predicted diagnosis based on your exact symptoms is: {diagnosis}"
        else:
            # No exact match found
            response = "No exact match found for the given symptoms."

        return response
    except Exception as e:
        logging.error(f"Error in predict_diagnosis: {str(e)}")
        return f"Error: {str(e)}"
def predict_future_diabetes(user_msg):
    try:
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()
        cursor.execute("SELECT age, bmi, blood_pressure, glucose, insulin, skin_thickness, diabetes_pedigree_function, pregnancies, diabetes, emp_id FROM diabetes_data")
        data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=column_names)
        connection.close()

        feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']
        X = df[feature_columns]
        y = df['diabetes']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression()
        model.fit(X_scaled, y)

        emp_id = extract_checkup_id(user_msg)
        if emp_id is not None and 'emp_id' in df.columns:
            emp_data = df[df['emp_id'] == emp_id][feature_columns]
            if not emp_data.empty:
                emp_data_scaled = scaler.transform(emp_data)
                prediction = model.predict_proba(emp_data_scaled)
                predicted_prob = prediction[:, 1].mean()

                if predicted_prob > 0.5:
                    risk_status = "High risk"
                    future_diabetes_patients = "Future diabetes patients"
                elif predicted_prob > 0.2:
                    risk_status = "Moderate risk"
                    future_diabetes_patients = "Future diabetes patients"
                else:
                    risk_status = "Low risk"
                    future_diabetes_patients = "Not future diabetes patients"

                response = f"Employee ID {emp_id}: Predicted probability - {predicted_prob:.2f}%, Status - {risk_status}, {future_diabetes_patients}"
            else:
                response = f"No data found for Employee ID {emp_id}."
        else:
            response = "Employee ID information is not available in the data."
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def predict_heart_disease(user_msg):
    try:
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()
        cursor.execute("SELECT age, cholesterol, blood_pressure, heart_rate, glucose, exercise, heart_disease, emp_id FROM diabetes_data")
        data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=column_names)
        connection.close()

        # Define feature columns
        feature_columns = ['age', 'cholesterol', 'blood_pressure', 'heart_rate', 'glucose', 'exercise']
        
        # Encoding categorical features
        categorical_features = ['exercise']
        numeric_features = ['age', 'cholesterol', 'blood_pressure', 'heart_rate', 'glucose']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ]
        )
        
        X = df[feature_columns]
        y = df['heart_disease']
        
        # Encode categorical feature
        X_encoded = preprocessor.fit_transform(X)
        
        # Train the model
        model = LogisticRegression()
        model.fit(X_encoded, y)
        
        # Extract Employee ID from user message
        emp_id = extract_checkup_id(user_msg)
        if emp_id is not None and 'emp_id' in df.columns:
            emp_data = df[df['emp_id'] == emp_id][feature_columns]
            if not emp_data.empty:
                emp_data_encoded = preprocessor.transform(emp_data)
                prediction = model.predict_proba(emp_data_encoded)
                predicted_prob = prediction[:, 1].mean()
                
                if predicted_prob > 0.5:
                    risk_status = "High risk"
                    future_heart_disease_patients = "Future heart disease patients"
                elif predicted_prob > 0.2:
                    risk_status = "Moderate risk"
                    future_heart_disease_patients = "Future heart disease patients"
                else:
                    risk_status = "Low risk"
                    future_heart_disease_patients = "Not future heart disease patients"
                
                response = f"Employee ID {emp_id}: Predicted probability - {predicted_prob:.2f}%, Status - {risk_status}, {future_heart_disease_patients}"
            else:
                response = f"No data found for Employee ID {emp_id}."
        else:
            response = "Employee ID information is not available in the data."
        return response
    except Exception as e:
        return f"Error: {str(e)}"
def predict_hypertension(user_msg):
    try:
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()
        cursor.execute("SELECT age, blood_pressure, cholesterol, glucose, bmi, heart_rate, hypertension, emp_id FROM diabetes_data")
        data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=column_names)
        connection.close()

        feature_columns = ['age', 'blood_pressure', 'cholesterol', 'glucose', 'bmi', 'heart_rate']
        X = df[feature_columns]
        y = df['hypertension']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression()
        model.fit(X_scaled, y)

        emp_id = extract_checkup_id(user_msg)
        if emp_id is not None and 'emp_id' in df.columns:
            emp_data = df[df['emp_id'] == emp_id][feature_columns]
            if not emp_data.empty:
                emp_data_scaled = scaler.transform(emp_data)
                prediction = model.predict_proba(emp_data_scaled)
                predicted_prob = prediction[:, 1].mean()

                if predicted_prob > 0.5:
                    risk_status = "High risk"
                    future_hypertension_patients = "Future hypertension patients"
                elif predicted_prob > 0.2:
                    risk_status = "Moderate risk"
                    future_hypertension_patients = "Future hypertension patients"
                else:
                    risk_status = "Low risk"
                    future_hypertension_patients = "Not future hypertension patients"

                response = f"Employee ID {emp_id}: Predicted probability - {predicted_prob:.2f}%, Status - {risk_status}, {future_hypertension_patients}"
            else:
                response = f"No data found for Employee ID {emp_id}."
        else:
            response = "Employee ID information is not available in the data."
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Function to count diabetes patients
def count_diabetes_patients(user_msg):
    try:
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()
        cursor.execute("SELECT age, bmi, blood_pressure, glucose, insulin, skin_thickness, diabetes_pedigree_function, pregnancies, diabetes, emp_id FROM diabetes_data")
        data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=column_names)
        connection.close()

        # Feature columns for prediction
        feature_columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']
        X = df[feature_columns]
        y = df['diabetes']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train logistic regression model
        model = LogisticRegression()
        model.fit(X_scaled, y)

        # Predict diabetes
        predictions = model.predict(X_scaled)
        df['predicted_diabetes'] = predictions

        # Count the number of predicted diabetes patients
        diabetes_count = df[df['predicted_diabetes'] == 1].shape[0]

        # Retrieve emp_ids of employees predicted to have diabetes
        emp_ids_with_diabetes = df[df['predicted_diabetes'] == 1]['emp_id'].tolist()

        # Format the response
        emp_ids_str = ", ".join(str(emp_id) for emp_id in emp_ids_with_diabetes)
        response = f"Number of predicted diabetes patients: {diabetes_count}%. Employees with diabetes: {emp_ids_str}"
        return response
    except Exception as e:
        return f"Error: {str(e)}"


def get_employee_data(emp_id, fields):
    try:
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()

        # Build the SELECT clause based on requested fields
        field_map = {
            'age': 'age',
            'height': 'height',
            'weight': 'weight',
            'blood_pressure': 'blood_pressure',
            'pulse': 'pulse'
        }
        
        selected_fields = [field_map[field] for field in fields if field in field_map]
        
        if not selected_fields:
            return "No valid fields requested."

        query = f"SELECT {', '.join(selected_fields)} FROM medical_examination WHERE emp_id = %s"
        cursor.execute(query, (emp_id,))
        data = cursor.fetchone()
        connection.close()

        if data:
            response = f"Employee ID {emp_id}:\n"
            response += "\n".join(f"{field.replace('_', ' ').title()}: {value}" 
                                  for field, value in zip(selected_fields, data))
        else:
            response = f"No data found for Employee ID {emp_id}."

        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Homepage route to serve HTML form
@app.route('/')
def index():
    return render_template('index.html', chat_history=[], complaints=[])

# Endpoint to handle user messages
@app.route('/send', methods=['POST'])


def handle_message():
    user_msg = request.form.get('msg', '').lower()
    selected_symptoms = request.form.getlist('selected_symptoms[]')
    response = ""

    logging.debug(f"Received user message: {user_msg}")
    logging.debug(f"Selected symptoms: {selected_symptoms}")

    if selected_symptoms:
        symptoms = {symptom: 1 for symptom in selected_symptoms}
        response = predict_diagnosis(symptoms)
    else:
        try:
            connection = mysql.connector.connect(**mysql_config)
            cursor = connection.cursor()

            # Check if the message already exists in the response_table
            cursor.execute("SELECT function_name, `key` FROM response_table WHERE user_message = %s", (user_msg,))
            result = cursor.fetchone()

            if result:
                function_name, key_value = result
                logging.debug(f"Function found in response table: {function_name}, key: {key_value}")

                # Call the function based on the function name
                response = call_function_based_on_name(function_name, user_msg, key_value)  # Ensure key_value is passed

            else:
                # Tokenize the user message
                doc = nlp(user_msg)
                tokens = [token.text for token in doc]
                logging.debug(f"Tokens extracted from user message: {tokens}")

                # Fetch all keyword-function mappings
                cursor.execute("SELECT keyword, function_name, `key` FROM keyword_functions")
                all_keywords = cursor.fetchall()

                function_scores = {}

                for row in all_keywords:
                    keyword_list = row[0].split(',')
                    function_name = row[1]
                    key = row[2]

                    # Check if all keywords in the list are in the user message tokens
                    if all(keyword in tokens for keyword in keyword_list):
                        logging.debug(f"All keywords matched for function: {function_name}")
                        
                        if function_name in function_scores:
                            function_scores[function_name]['score'] += len(keyword_list)
                        else:
                            function_scores[function_name] = {'score': len(keyword_list), 'key': key}

                logging.debug(f"Function match scores: {function_scores}")

                if function_scores:
                    # Select the best match based on the highest score
                    best_match_function = max(function_scores, key=lambda fn: function_scores[fn]['score'])
                    logging.debug(f"Best match function: {best_match_function}")

                    key_value = function_scores[best_match_function]['key']

                    # Handle special cases for 'handle_disease' and 'handle_keyword'
                    if best_match_function in ['handle_disease', 'handle_keyword']:
                        response = globals()[best_match_function](best_match_function, key_value)  # Call the specific function
                    else:
                        response = call_function_based_on_name(best_match_function, user_msg, key_value)  # Other functions

                    # Insert matched response into the response table to avoid recalculating later
                    cursor.execute("INSERT IGNORE INTO response_table (user_message, function_name, `key`) VALUES (%s, %s, %s)",
                                   (user_msg, best_match_function, key_value))
                    connection.commit()
                else:
                    response = "I couldn't find a relevant keyword to process your request."

        except mysql.connector.Error as err:
            logging.error(f"Database error: {err}")
            response = "Error connecting to the database."
        finally:
            if connection.is_connected():
                connection.close()

    logging.debug(f"Response generated: {response}")
    return jsonify({'response': response})

# Utility function to call the appropriate function based on the function name
def call_function_based_on_name(function_name, user_msg, key_value):
    if function_name == "predict_future_diabetes":
        return predict_future_diabetes(user_msg)
    elif function_name == "predict_heart_disease":
        return predict_heart_disease(user_msg)
    elif function_name == "predict_hypertension":
        return predict_hypertension(user_msg)
    elif function_name == "get_employee_data":
        emp_id = extract_checkup_id(user_msg)
        fields = [field for field in ['age', 'height', 'weight', 'blood_pressure', 'pulse'] if field in user_msg]
        return get_employee_data(emp_id, fields)
    elif function_name == "count_diabetes_patients":
        return count_diabetes_patients(user_msg)
    elif function_name in ['handle_disease', 'handle_keyword']:
        # Pass key_value as an argument to the global function call
        return globals()[function_name](function_name, key_value)
    else:
        return "No matching function found."

if __name__ == '__main__':
    app.run(debug=True)
