import os
import subprocess
import psycopg2
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Read the environment variables
# Configuration for your database
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

# Check if the Results table is empty
def check_database(DB_NAME, DB_USER, DB_PASS, DB_HOST, DB_PORT):
    """
    check if there are data in the database table
    Run populate_database.py if database is empty
    """
    
    conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)

    
    check_query = 'SELECT COUNT(*) FROM "Results";'
    cursor = conn.cursor()
    cursor.execute(check_query)
    count = cursor.fetchone()[0]

    # If the table is empty, run the populate_database
    if count == 0:
        print('Database empty!')
        #print('Populating Database')
        #subprocess.run(["python3", 'populate_database.py'])
    else:
        print(f'\nDatabase initiated\n')

# check database before intiating database
check_database(DB_NAME, DB_USER, DB_PASS, DB_HOST, DB_PORT)