FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app/

# Install dependencies
RUN pip install -r requirements.txt

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]