# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8888 available to the world outside this container (for Jupyter)
EXPOSE 8888

# Define environment variable
ENV NAME LottoPredictor
ENV PYTHONPATH /app

# Run app.py when the container launches
CMD ["python", "src/main.py"]
