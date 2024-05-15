from flask import Flask, request, jsonify
from flasgger import Swagger
from roster_model.src.main import *

app = Flask(__name__)
Swagger = Swagger(app)

@app.route('/ml_model/training', methods=['POST'])
def train():
    """
    Train the ML Model
    ---
    responses:
      200:
        description: Returns a message indicating the model training status.
    """

    roster = RosterModel(DB_NAME, DB_USER, DB_PASS, DB_HOST, DB_PORT)
    roster.train_model(schedule_training=False)

    return jsonify({"message": "Model trained successfully"})

@app.route('/ml_model/predict', methods=['POST'])
def assign_shift():
    """
    Assign Shift using the ML Model
    ---
    responses:
      200:
        description: Returns a message indicating the shift assignment status.
    """
    
    roster = RosterModel(DB_NAME, DB_USER, DB_PASS, DB_HOST, DB_PORT)
    roster.assign_shift()

    return jsonify({"message": "Shift assigned"})

if __name__ == '__main__':
    add_cron_jobs()
    # Intiate the api
    app.run(host='0.0.0.0', port=5000, use_reloader = False)