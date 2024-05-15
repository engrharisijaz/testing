# Start with the latest Ubuntu image as the base
FROM ubuntu:latest

# Update and install basic tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    vim \
    git \
    curl \
    wget \
    openjdk-8-jdk \
    # Add other tools you need for your project
    && rm -rf /var/lib/apt/lists/*


# Add user and password 
RUN useradd -m test && echo "test:test" | chpasswd


# Set the PYTHONPATH
ENV PYTHONPATH /app:$PYTHONPATH

# Set the working directory to /app
WORKDIR /app

# Copy your scripts into the Docker image
COPY . /app/roster_ml

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt
