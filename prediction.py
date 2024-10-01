from flask import Flask, request, jsonify, render_template
import mysql.connector
import spacy
import logging
import pandas as pd
import difflib

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

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

def extract_checkup_id(user_msg):
    for word in user_msg.split():
        if word.isdigit():
            return int(word)
    return None

def load_data_from_excel():
    try:
        return pd.read_excel('symptom_diagnosis_data.xlsx')
    except Exception as e:
        logging.error(f"Error loading data from Excel: {str(e)}")
        return pd.DataFrame()

def load_data():
    try:
        connection = mysql.connector.connect(**mysql_config)
        df = pd.read_sql("SELECT * FROM medical_data_table", connection)
        connection.close()
        return df
    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
        return pd.DataFrame()

def predict_diagnosis(symptoms):
    try:
        df = load_data_from_excel()
        if df.empty:
            return "Error loading data."

        feature_columns = [col for col in df.columns if col != 'diagnosis']
        df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce').dropna()

        if 'diagnosis' not in df.columns:
            return "Error: 'diagnosis' column is missing in the data."

        symptoms_dict = {col: 0 for col in feature_columns}
        for symptom in symptoms:
            matched_column = difflib.get_close_matches(symptom, feature_columns, n=1, cutoff=0.8)
            if matched_column:
                symptoms_dict[matched_column[0]] = 1

        symptoms_series = pd.Series(symptoms_dict)
        df_aligned, symptoms_aligned = df[feature_columns].align(symptoms_series, axis=1, fill_value=0)
        matching_row = df[df_aligned.eq(symptoms_aligned, axis=1).all(axis=1)]

        if not matching_row.empty:
            return f"The predicted diagnosis based on your exact symptoms is: {matching_row['diagnosis'].values[0]}"
        else:
            return "No exact match found for the given symptoms."

    except Exception as e:
        logging.error(f"Error in predict_diagnosis: {str(e)}")
        return f"Error: {str(e)}"

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

@app.route('/')
def index():
    return render_template('index.html', chat_history=[], complaints=[])

@app.route('/send', methods=['POST'])
def handle_message():
    user_msg = request.form.get('msg', '').lower()
    selected_symptoms = request.form.getlist('selected_symptoms[]')
    response = ""

    if selected_symptoms:
        response = predict_diagnosis(selected_symptoms)
    else:
        try:
            connection = mysql.connector.connect(**mysql_config)
            cursor = connection.cursor()

            doc = nlp(user_msg)
            tokens = [token.text for token in doc]
            logging.debug(f"Tokens extracted from user message: {tokens}")

            cursor.execute("SELECT keyword, function_name, `key` FROM keyword_functions")
            keyword_functions = cursor.fetchall()
            keywords = [row[0] for row in keyword_functions]
            functions = {row[0]: row[1] for row in keyword_functions}

            for keyword in keywords:
                if keyword in tokens:
                    key = next((row[2] for row in keyword_functions if row[0] == keyword), None)
                    if key:
                        response = handle_keyword(keyword, key)
                        break
            if not response:
                response = "Keyword not recognized or no corresponding function found."

        except mysql.connector.Error as err:
            logging.error(f"Database error: {err}")
            response = "Error retrieving keyword functions."

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
