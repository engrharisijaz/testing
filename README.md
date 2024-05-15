# Documentation for Running the Roster Model Application

The Roster Model application is designed to train machine learning models and assign shifts to users. It leverages a PostgreSQL database to store and manage data and provides API endpoints to interact with the model. This documentation provides steps on how to set up, run, and test the application.

## Pre-requisites:
- Docker installed on your machine.
- Docker Compose installed on your machine.

## Steps to Run the Application:

1. **Clone the Repository** (if applicable):
    - Clone the repository containing the Docker Compose file and the necessary scripts to your local machine.

2. **Build and Start the Services**:
    - Navigate to the directory containing the `docker-compose.yml` file.
    - Run the following command to build and start the services defined in the `docker-compose.yml` file:
        ```bash
        docker-compose up --build -d
        ```

3. **Interact with the API**:
    - Once the services are up, the Flask application's API endpoints will be accessible. The api swagger documnetation can be found here.
    ```
    http://{server_name}:ports/apidocs/
    ```
    in this case the documentation can be accessed through the browser at 
    ```
    http://localhost:5000/apidocs/
    ```
    The default port id 5000, you can update it in the docker-compose file 
    - To train the model, send a POST request to:
        ```
        http://{server_name}:5000/ml_model/training
        ```
        In this case the endpoint can be accessed at the following endpoint where the {server name} is localhost
        ```
        http://localhost:5000/ml_model/training
        ```

    - To assign shifts to users, send a POST request to:
        ```
        http://{server_name}:5000/ml_model/predict
        ```
        In this case the endpoint can be accessed at the following endpoint where the {server name} is localhost
        ```
        http://localhost:5000/ml_model/predict
        ```

4. **Run Unit Tests**:
    - To ensure that the application is working as intended, you can run the unit tests provided.
    - Attach a shell to the `ubuntu_dev` container with the command:
        ```bash
        docker exec -it roster_ml_ubuntu_dev bash
        ```
    - Once inside, navigate to the test directory.
    - Run the tests with the following command:
        ```bash
        pytest --cov=roster_model tests/
        ```
    - Note: The tests have a coverage of 97%, which surpasses the best practice benchmark of 85%.

5. **Access PGAdmin** (Optional):
    - You can access PGAdmin through your web browser at `http://localhost`.
    - Login with the email `user@domain.com` and password `SuperSecret` to manage your PostgreSQL database.

6. **Stop the Services** (when done):
    - Exit the `ubuntu_dev` container shell.
    - Run the following command to stop and remove all running containers defined in the `docker-compose.yml` file:
        ```bash
        docker-compose down
        ```
7. **Running the flask app directly** (Optional):
    - Create a virtual Environment and activate it. 
        Create env command: 
            ```
                python -m venv .venv
            ```
        Activate env command:
            ```
                ./.venv/Scripts/activate
            ```
    - Run following command to install requirements 
        ```
            pip install -r requirements.txt
        ```
    - Run the following command to run the flask app directly (for debugging)
        ```
            python3 app.py
        ```
8. **Running Program in detach mode in AWS EC2 instance**
    - To run the program in detach mode in AWS EC2 instance, you can use the following command
        - Run new terminal
        - In new terminal run the following command
        ```
            Screen -- 
        ```
        - Run the following command to start the new terminal later it will be detached
        - Run the flask server using
        ```
            python3 app.py (consider repeating step 7 to activate the virtual environment and install requirements)
        ```
        - Detach the screen by pressing `Ctrl + A + D`
        - To reattach the screen run the following command
        ```
            screen -r
        ```
        - To exit the screen press `Ctrl + D`
        - To kill the screen run the following command
        ```
            screen -X -S [session # you want to kill] quit
        ```
        - To list the screen run the following command
        ```
            screen -ls
        ```
        