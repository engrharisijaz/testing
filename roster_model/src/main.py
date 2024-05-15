import os
import pandas as pd
import numpy as np
import tensorflow as tf
import psycopg2
from psycopg2 import sql
from flask_apscheduler import APScheduler
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from roster_model.src.initiate_database import * # initiate_database.py -> check_database() will automatically run

# Constants
CLASS_COUNT_FILE = 'roster_model/class_count.txt'

scheduler = APScheduler()

def save_class_count(class_count):
    with open(CLASS_COUNT_FILE, 'w') as f:
        f.write(str(class_count))

def get_last_class_count():
    if os.path.exists(CLASS_COUNT_FILE):
        with open(CLASS_COUNT_FILE, 'r') as f:
            return int(f.read())
    return None

def schedule_retraining(): # Function reponsible for scheduling th re-training process
    try:
      roster = RosterModel(DB_NAME, DB_USER, DB_PASS, DB_HOST, DB_PORT)
      roster.train_model(schedule_training=True) # calling the training function for re training 
    except Exception as e:
        print(e)

def add_cron_jobs(): # Cron jobs scheduling functions
    # Retraining ID: retraining_ + Current date is used as the ID for the retraining cron job. Can set up a time according to the time zone on which the function will be triggered
    current_time = str("retraining_" + str(datetime.now()))
    scheduler.add_job(current_time, func=schedule_retraining, trigger="cron", day_of_week="mon", timezone='US/Eastern', hour=6, minute = 0)
    scheduler.start() # starting added jobs

class RosterModel():
    def __init__(self, DB_NAME, DB_USER, DB_PASS, DB_HOST, DB_PORT):
        # Connect to your database
        self.conn = psycopg2.connect(
            database=DB_NAME, 
            user=DB_USER, 
            password=DB_PASS, 
            host=DB_HOST, 
            port=DB_PORT)

    def prep_train_data(self, schedule_training):
        """
        prep data for training
        """
        # Connect to the database
        cursor = self.conn.cursor()

        post_query = ""
        if schedule_training: # If schedule_training is True, then the training data will be for the last 1 week
            post_query = " AND r.updated_at > current_timestamp - interval '1 week';"
        
        # Define your SQL query with the date condition
        query = f"""
            SELECT 
                r.schedule_id, sch.shift_id, r.job_id,
                r.assigned_to, s.location_id, s.start_time, 
                s.end_time
            FROM 
                "Results" r
            LEFT JOIN 
                "Schedules" sch ON r.schedule_id = sch.id
            LEFT JOIN 
                "Shifts" s ON sch.shift_id = s.id
            WHERE 
                r.assigned_to IS NOT NULL
                {post_query}
        """

        # Execute the query
        cursor.execute(query)

        # Fetch the results
        results = cursor.fetchall()
        # Get the column names from the cursor description
        columns = [desc[0] for desc in cursor.description]

        # Create a Pandas DataFrame from the results
        self.train_df = pd.DataFrame(results, columns=columns)

        if self.train_df.empty:
            self.x_train = []
            self.y_train = []
            print("Empty DataFrame. Returning.")
            return

        # Calculate hours and convert to numeric
        self.train_df['hours'] = (self.train_df['end_time'] - self.train_df['start_time'])
        self.train_df['hours'] = self.train_df['hours'].dt.components['hours']
        
        self.y_train = self.train_df['assigned_to']

        def clean(df):
            df = df.drop(columns=[
                'start_time', 'end_time', 
                 'assigned_to',
                ], axis= 1)
            return df    

        self.x_train = clean(self.train_df)
        

        # Store number of available users used for training
        # Save the current class count after training
        # Same query (no change)
        query_number_of_users = """
                                SELECT 
                                    MAX(id)
                                FROM 
                                    "Users";
                                """
        cur = self.conn.cursor()
        cur.execute(query_number_of_users)
        result = cur.fetchone()            

        # Save the number of current total users available
        save_class_count(result[0])    

        self.num_of_class = result[0]

    def prep_pred_data(self):
        """
        prep data from prediction
        """
        try:
            # Query to fetch data from the database
                   
            # New Query
            query = """
                SELECT 
                    r.Id, r.schedule_id, sch.shift_id, r.job_id,
                    r.status, r.approved, r.rating,
                    s.location_id, s.start_time, 
                    s.end_time, s.created_at
                FROM 
                    "Results" r
                LEFT JOIN 
                    "Schedules" sch ON r.schedule_id = sch.id
                LEFT JOIN 
                    "Shifts" s ON sch.shift_id=s.id
                WHERE 
                    r.assigned_to IS NULL;
                    """
            # Load data into a DataFrame
            self.pred_df = pd.read_sql(query, self.conn)

            self.pred_df['hours'] = self.pred_df['end_time'] - self.pred_df['start_time']
            self.pred_df['hours'] = self.pred_df['hours'].dt.components['hours']

            self.prediction_df = self.pred_df.copy()
            
            def clean(df):
                df = df.drop(columns=[
                    'rating', 'approved', 
                    'start_time', 'end_time', 
                    'created_at', 'id', 
                    'status'
                    ], axis= 1)
                return df    

            self.x_pred = clean(self.prediction_df)
        except Exception as e:
            # Handle any error that occurs
            print(f"An error occurred: {e}")

    def train_model(self, schedule_training=False):
            self.prep_train_data(schedule_training)

            if schedule_training:
                model = keras.models.load_model("./trained_roster_model.keras")
                for layer in model.layers:
                    layer.trainable = True
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            else:
                model = Sequential(layers=[
                    layers.Input(shape=(self.x_train.shape[1],), dtype='float32'),
                    layers.Dense(256, activation='relu'),
                    layers.Dense(512, activation='relu'),
                    layers.Dense(self.num_of_class + 1, activation='softmax')
                ])
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

            if len(self.x_train) > 10:
                print('Training model')
                model.fit(self.x_train, self.y_train, epochs=10, validation_split=0.2)
                print('Training completed')
            else:
                print("There is not enough new data (previous one week data) in the database to train the model")
                return

            model.save('./trained_roster_model.keras')

    def assign_shift(self): # Main code for prediction
        self.prep_pred_data() # prepare self.x_pred

        if self.prediction_df.empty:
            print("No new shift to assign was found in database.")

        # check if there is a new user added to the users table
        # Retrain the model if there is a new user else use the current trained model
        # Same query(no change)
        query_number_of_users = """
                                    SELECT 
                                        MAX(id)
                                    FROM 
                                        "Users";
                                    """
        cur = self.conn.cursor()
        cur.execute(query_number_of_users)
        result = cur.fetchone()  

        # If there is a new user; retrain
        if result[0] is None or result[0] != get_last_class_count():
            self.train_model()

        model = tf.keras.models.load_model('./trained_roster_model.keras')

        # After training, use the model to make predictions
        predictions = model.predict(self.x_pred)

        # Define a threshold
        threshold = 0.0005

        # Create a dictionary to keep track of assignments
        assignments_dict = {}

        # Prepare a DataFrame to update with assigned classes
        x_pred_df = pd.DataFrame(self.x_pred)
        x_pred_df['id'] = self.prediction_df['id']  # Include the 'id' column for updating the database later


        cur = self.conn.cursor()

        def check_certifications(user_id, job_id):
            # Fetch the required certifications for the job
            # Same query(no change)
            cur.execute("""
                SELECT certifications
                FROM "Jobs"
                WHERE id = %s;
            """, (int(job_id),))  # convert numpy.int64 to int
            
            required_certifications = cur.fetchone()[0]

            # Remove curly braces and split by comma
            elements = required_certifications.strip().strip('{}').split(',')

            # Convert each string element to an integer
            required_certifications = [int(element) for element in elements]
            
            # Fetch the certifications the user has

            # Previous Query
            # cur.execute("""
            #     SELECT certification_id
            #     FROM "Certifications_Expires"
            #     WHERE "User_id" = %s;
            # """, (int(user_id),))  # convert numpy.int64 to int
            
            # New Query
            cur.execute("""
                SELECT certification_id
                FROM "Certifications_Expires"
                WHERE user_id = %s;
            """, (int(user_id),))  # convert numpy.int64 to int
            
            user_certifications = [cert[0] for cert in cur.fetchall()]
            
            # Check if the user has all required certifications
            return any(cert in user_certifications for cert in required_certifications)

        def get_user_info(user_ids):
            user_ids_list = user_ids.tolist()  # Convert numpy array to list
            # Same query(no change)
            user_info_query = """
                SELECT id, experience, pay_rate 
                FROM "Users"
                WHERE id = ANY(%s);
            """
            cur.execute(user_info_query, (user_ids_list,))
            user_info = cur.fetchall()
            return {info[0]: {'experience': info[1], 'pay_rate': info[2]} for info in user_info}

        def experience_to_numeric(level):
            mapping = {
                'Entry level': 1,
                'Experienced': 2,
                'Senior level': 3
            }
            return mapping.get(level, 0)  

        def check_shift_assignment(user_id, date):
            # New Query
            query = sql.SQL("""
                SELECT COUNT(*)
                FROM "Results" r
                JOIN "Schedules" sch ON r.schedule_id = sch.id
                JOIN "Shifts" s ON sch.shift_id = s.id
                WHERE r.assigned_to = %s AND DATE(s.start_time) = %s;
                """)

            cur.execute(query, (int(user_id), date))
            result = cur.fetchone()
            return result[0] > 0


        # Iterate through each row of predictions
        for idx, row in enumerate(predictions):
            # Get indices of classes that exceed the threshold
            candidate_indices = np.where(row > threshold)[0]
            
            if len(candidate_indices) == 0:
                x_pred_df.loc[idx, 'assigned_to'] = None  # No user available
                continue

            # Fetch user information for the current batch of candidate users
            user_info = get_user_info(candidate_indices)

            # Sort candidate users based on experience (descending) and pay rate (ascending)
            sorted_candidates = sorted(
            candidate_indices, # first priority is experience (3 highest priority ; 0 lowest priority)      second priority pay_rate
            key=lambda x: (-experience_to_numeric(user_info.get(x, {}).get('experience', 'Experienced')), float(user_info.get(x, {}).get('pay_rate', 0)))  )
            
            sorted_candidates = [candidate for candidate in sorted_candidates if candidate != 0]

            
            job_id = self.prediction_df.loc[idx, 'job_id']
            current_date = pd.to_datetime(self.prediction_df.loc[idx, 'start_time']).date()

            assigned = False
            for candidate in sorted_candidates:
                user_id = candidate
                
                # Check if user is already assigned a shift on this date
                if check_shift_assignment(user_id, current_date):
                    continue  # If the user has a shift on this date, skip to the next candidate (this check will be performed on db table)

                key = (current_date, user_id)
                if key in assignments_dict:
                    continue # if user is assigned to any shift in current bulk then skip to the next user

                # Check user's certifications
                if not check_certifications(user_id, job_id):
                    continue  # If the user doesn't have required certifications, skip to the next candidate
                
                # If the user passes both checks, assign them to the shift
                x_pred_df.loc[idx, 'assigned_to'] = user_id
                assignments_dict[key] = True
                assigned = True
                break  # Exit loop once a suitable candidate is found
            
            if not assigned:
                x_pred_df.loc[idx, 'assigned_to'] = None  # No suitable candidate was found


        cur = self.conn.cursor()

        # Iterate through the DataFrame and update the Results table
        for index, row in x_pred_df.iterrows():
            # Skip rows where 'assigned_to' is NaN or not a number
            if pd.isna(row['assigned_to']) or not np.isfinite(row['assigned_to']):
                continue
            
            try:
                # Same query(no change)
                query = sql.SQL(
                    """
                    UPDATE "Results"
                    SET assigned_to = %s
                    WHERE id = %s;
                    """
                )
                cur.execute(query, (int(row['assigned_to']), int(row['id'])))
            except ValueError as ve:
                continue  # Skip to the next row if a ValueError occurs

        # Commit the changes and close the connection
        self.conn.commit()
        cur.close()
        self.conn.close()

        print('New shift has been assigned to available users')