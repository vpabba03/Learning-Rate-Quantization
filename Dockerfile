FROM python:3.11

# Set the working directory to /app (or another appropriate location)
WORKDIR /app

# Install virtualenv (to create virtual environments)
RUN pip install virtualenv

# Create a virtual environment named 'venv'
RUN python -m venv /app/venv

# Set environment variable to use the virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Copy over the requirements for the Docker container
COPY requirements.txt /app/

# Install the requirements into the virtual environment
RUN pip install -r /app/requirements.txt

# Copy over the scripts
COPY ./src/ /src

# Set the working directory to /src
WORKDIR /src