import logging
import mysql.connector
from config import mysql_config
from prediction import predict_future_diabetes, get_employee_data, handle_keyword

# Extract checkup ID from user message
def extract_checkup_id(user_msg):
    words = user_msg.split()
    for word in words:
        if word.isdigit():
            return int(word)
    return None
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
def handle_keyword(user_msg):
    connection = mysql.connector.connect(**mysql_config)
    cursor = connection.cursor()

    doc = nlp(user_msg)
    tokens = [token.text for token in doc]
    logging.debug(f"Tokens extracted from user message: {tokens}")
    keyword_found = False
    cursor.execute("SELECT keyword, function_name, `key` FROM keyword_functions")
    all_keywords = cursor.fetchall()

    logging.debug(f"Fetched keywords from database: {all_keywords}")

    for token in tokens:
        logging.debug(f"Processing token: {token}")
        for row in all_keywords:
            keyword_list = row[0].split(',')
            logging.debug(f"Checking against keyword list: {keyword_list}")
            if token in keyword_list:
                keyword = token
                function_name = row[1]
                key = row[2]
                logging.debug(f"Keyword matched: {keyword}, function_name: {function_name}, key: {key}")

                if function_name == "predict_future_diabetes":
                    response = predict_future_diabetes(user_msg)
                elif function_name == "get_employee_data":
                    emp_id = extract_checkup_id(user_msg)
                    fields = [field for field in ['age', 'height', 'weight', 'blood_pressure', 'pulse'] if field in user_msg]
                    response = get_employee_data(emp_id, fields)
                else:
                    response = handle_keyword(keyword, key)

                keyword_found = True
                break
        if keyword_found:
            break

    if not keyword_found:
        response = "I couldn't find a relevant keyword to process your request."

    connection.close()
    return response
