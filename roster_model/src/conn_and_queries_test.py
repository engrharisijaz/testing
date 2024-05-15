
import psycopg2
import pandas as pd

conn = psycopg2.connect(database='New_Database_RosterML',
                        user='postgres', password='saeed50ajmal', host='localhost', port=5432)

if conn:
    print("\nConnection Seccessful !\n")

cur = conn.cursor()

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
            "Shifts" s ON sch.shift_id = s.id
        WHERE 
            r.assigned_to IS NULL;
        """

cur.execute(query)

print(cur.fetchone())

# Load data into a DataFrame
pred_df = pd.read_sql(query, conn)

pred_df['hours'] = pred_df['end_time'] - pred_df['start_time']
pred_df['hours'] = pred_df['hours'].dt.components['hours']

prediction_df = pred_df.copy()

def clean(df):
    df = df.drop(columns=[
        'rating', 'approved', 
        'start_time', 'end_time', 
        'created_at', 'id', 
        'status'
        ], axis= 1)
    return df    

x_pred = clean(prediction_df)
print(x_pred)

#---------------------------------------------------------------------------------------------

query_number_of_users = """
                        SELECT 
                            MAX(id)
                        FROM 
                            "Users";
                        """

cur = conn.cursor()
cur.execute(query_number_of_users)
result = cur.fetchone()
print('\nquery_number_of_users: ', result[0])

#---------------------------------------------------------------------------------------------

query3 = """
        SELECT certifications
        FROM "Jobs"
        WHERE id = 1;
        """
cur = conn.cursor()
cur.execute(query3)
required_certifications = cur.fetchone()[0]

elements = required_certifications.strip().strip('{}').split(',')

required_certifications = [int(element) for element in elements]
print(f'\nrequired_certifications: {required_certifications}')

#---------------------------------------------------------------------------------------------

query4 = """
        SELECT certification_id
        FROM "Certifications_Expires"
        WHERE user_id = 1;
        """

cur = conn.cursor()
cur.execute(query4)
user_certifications = [cert[0] for cert in cur.fetchall()]
print(f'user_certifications: {user_certifications}')
print('Return:', any(cert in user_certifications for cert in required_certifications))

#---------------------------------------------------------------------------------------------

query5 = """
        SELECT id, experience, pay_rate 
        FROM "Users"
        WHERE id = ANY(%s);
        """
cur = conn.cursor()
cur.execute(query5, ([1, 263],))
user_info = cur.fetchall()
print('Return:', {info[0]: {'experience': info[1], 'pay_rate': info[2]} for info in user_info})

#---------------------------------------------------------------------------------------------

query6 = """
        SELECT COUNT(*)
        FROM "Results" r
        JOIN "Schedules" sch ON r.schedule_id = sch.id
        JOIN "Shifts" s ON sch.shift_id = s.id
        WHERE r.assigned_to = %s AND DATE(s.start_time) = %s;
        """

cur = conn.cursor()
cur.execute(query6, (1, pd.to_datetime("2023-10-16 21:00:00+05").date()))
result = cur.fetchone()
print(result)
print("Return: ", result[0] > 0)