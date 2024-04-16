# Use the official Python image from the Docker Hub
FROM python:3.9-slim


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container at /app
COPY . /app/

CMD ["streamlit", "run", "app.py"]
