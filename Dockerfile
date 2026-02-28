# 1. Use the exact Python version from your virtual environment
FROM python:3.14-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install the system library required for XGBoost and LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Only copy the specific folders and files needed to run the app!
# This safely ignores venv, notebooks, logs, and .ebextensions
COPY artifacts/ ./artifacts/
COPY src/ ./src/
COPY templates/ ./templates/
COPY application.py .
COPY setup.py .

# 6. Set Flask environment variables
ENV FLASK_APP=application.py

# 7. Expose the port Hugging Face requires
EXPOSE 7860

# 8. Command to run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]