FROM python:3.10-slim

# Install Git
RUN apt-get update && apt-get install -y git

# Set the working directory
WORKDIR /workspace

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run the application
CMD ["python", "src/main.py"]