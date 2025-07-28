# Use Python 3.11 (compatible with transformers and torch)
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y gcc g++ git wget && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model
RUN python -m spacy download en_core_web_sm

# Copy your main script
COPY persona_analysis_graphrag.py .

# Create input/output folders
RUN mkdir -p /app/input /app/output

# Default command
CMD ["python", "persona_analysis_graphrag.py"]
