# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install flask pandas numpy

# Expose the port
EXPOSE 5000

# Run the application
CMD ["python", "deploy/app.py"]

