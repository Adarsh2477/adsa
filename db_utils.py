import mysql.connector
import pandas as pd
from config import mysql_config

def load_data():
    connection = mysql.connector.connect(**mysql_config)
    query = "SELECT * FROM medical_data_table"
    df = pd.read_sql(query, connection)
    connection.close()
    return df
