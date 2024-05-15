import pytest
import pandas as pd
from flask import json
from datetime import datetime
from unittest.mock import patch, MagicMock, Mock
from roster_model.src.main import RosterModel
from app import app

# Mock the database connection
@pytest.fixture
def mock_db_connection():
    with patch('psycopg2.connect') as mock_conn:
        mock_conn.return_value = MagicMock()
        yield mock_conn


# Create a pytest fixture that provides mock data
@pytest.fixture
def mock_query_data():
    # Mock data for "Results" table
    results_data = {
        "Id": [1, 2, 3, 4, 5],
        "schedule_id": [1, 2, 3, 4, 5],
        # "shift_id": [1, 2, 3, 4, 5],
        "job_id": [1, 2, 3, 4, 5],
        "assigned_to": [1, 2, None, None, None],
        "status": ["completed", "completed", "Pending", "Pending", "Pending"],
        "approved": [True, True, False, False, False],
        "rating": [4, 5, 3, None, None]
    }
    results_df = pd.DataFrame(results_data)

    # Mock data for "Shifts" table
    shifts_data = {
        "id": [1, 2, 3, 4, 5],
        "location_id": [1, 2, 3, 4, 5],
        "start_time": [datetime(2023, 1, 1, 9, 0), datetime(2023, 1, 2, 9, 0), datetime(2023, 1, 3, 9, 0), datetime(2023, 1, 4, 9, 0), datetime(2023, 1, 5, 9, 0)],
        "end_time": [datetime(2023, 1, 1, 17, 0), datetime(2023, 1, 2, 17, 0), datetime(2023, 1, 3, 17, 0), datetime(2023, 1, 4, 17, 0), datetime(2023, 1, 5, 17, 0)],
        "created_at": [datetime(2023, 1, 1, 8, 0), datetime(2023, 1, 2, 8, 0), datetime(2023, 1, 3, 8, 0), datetime(2023, 1, 4, 8, 0), datetime(2023, 1, 5, 8, 0)]
    }
    shifts_df = pd.DataFrame(shifts_data)

    # Mock data for "Schedules" table
    schedules_data = {
        "id": [1, 2, 3, 4, 5],
        "shift_id": [1, 2, 3, 4, 5]
    }
    schedules_df = pd.DataFrame(schedules_data)

    results_df = results_df.merge(schedules_df, left_on = "schedule_id", right_on = "id")
    results_df.drop(columns=['id'], inplace=True)

    mock_query_data = pd.merge(results_df, shifts_df, left_on="shift_id", right_on="id", how="left")

    return mock_query_data


# Fixture for mock database connection
@pytest.fixture
def mock_db_connection():
    with patch('psycopg2.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        # This will simulate the execution of a SQL command without actually interacting with a real database.
        mock_cur.execute.return_value = None
        mock_cur.fetchone.return_value = None
        mock_cur.fetchall.return_value = []
        mock_connect.return_value = mock_conn
        yield mock_conn

@pytest.fixture
def client():
    return app.test_client()

def test_some_feature(mock_query_data):
    # The mock_data fixture provides a DataFrame similar to the query results
    assert len(mock_query_data) == 5

def test_prep_data(mock_db_connection, mock_query_data):
    roster = RosterModel("testdb", "testuser", "testpass", "testhost", "testport")
    
    # Patch the pd.read_sql function to return the mock_query_data
    with patch('pandas.read_sql', return_value=mock_query_data):
        roster.prep_train_data()
    
    # Assertions
    assert roster.train_df is not None
    assert 'hours' in roster.train_df.columns
    assert 'assigned_to' in roster.train_df.columns


@pytest.fixture
def mock_db_connection():
    with patch('psycopg2.connect') as mock_connection:
        mock_cursor = MagicMock()
        mock_connection.return_value.cursor.return_value = mock_cursor

        # Set the predefined value for the 'fetchone' method
        mock_cursor.fetchone.return_value = [10]  # Example value for 'self.num_of_class'
        yield mock_connection

        
def test_train_model(mock_db_connection, mock_query_data):
    roster = RosterModel("testdb", "testuser", "testpass", "testhost", "testport")
    
    # Patch the pd.read_sql function to return the mock_query_data
    with patch('pandas.read_sql', return_value=mock_query_data):
        with patch('tensorflow.keras.Sequential.fit') as mock_fit:

            roster.prep_train_data()
            roster.train_model()
    
    # Assertions
    mock_fit.assert_called_once()


# Helper function to provide results for fetchone and fetchall
def side_effect(cursor_method, query, params=None):
    if "Jobs" in query:
        # Modify this to return what you would expect from the Jobs query
        return ([1, 2, 3],)
    if "Certifications_Expires" in query:
        # Modify this to return what you would expect from the Certifications_Expires query
        return [(1,), (2,), (3,)]
    if "Results" in query and "Shifts" in query:
        # Modify this based on what you expect from this query
        return [(1,), (2,), (3,)]
    # Add more conditions as needed
    return None

# Mock for the database connection
def test_assign_shift(mock_query_data):  # Add the fixture as an argument to the test function
    # Mock for the database connection
    mocked_conn = Mock()
    mocked_cursor = mocked_conn.cursor.return_value
    mocked_cursor.fetchone.side_effect = lambda: side_effect('fetchone', mocked_cursor.execute.call_args[0][0])
    mocked_cursor.fetchall.side_effect = lambda: side_effect('fetchall', mocked_cursor.execute.call_args[0][0])
    mocked_cursor.execute.side_effect = lambda query, params=None: None

    # Test for the assign_shift method
    @patch('psycopg2.connect', return_value=mocked_conn)
    @patch('pandas.read_sql', side_effect=mock_query_data)  # We'll continue to use the mock_query_data for mocking pd.read_sql
    def inner_test(mocked_connect, mocked_sql):
        # Create the RosterModel instance
        roster = RosterModel("testdb", "testuser", "testpass", "testhost", "testport")
        # Call the assign_shift method

        roster.assign_shift()

        mocked_sql.assert_called_once()


def test_train_model_api_endpoint(client):
    with patch('roster_model.src.main.RosterModel') as MockRosterModel:
        mock_instance = MockRosterModel.return_value
        mock_instance.train_model.return_value = None

        response = client.post('/ml_model/training')

        assert response.status_code == 200
        data = json.loads(response.data.decode('utf-8'))
        assert data["message"] == "Model trained successfully"

def test_assign_shift_api_endpoint(client):
    with patch('roster_model.src.main.RosterModel') as MockRosterModel:
        mock_instance = MockRosterModel.return_value
        mock_instance.assign_shift.return_value = None

        response = client.post('/ml_model/predict')

        assert response.status_code == 200
        data = json.loads(response.data.decode('utf-8'))
        assert data["message"] == "Shift assigned"  
