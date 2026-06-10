FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install uv, then use it to install dependencies from pyproject.toml
RUN pip install uv && uv pip install --system -r pyproject.toml

# Default command — swap out for whichever script you want to run
CMD ["python", "bio_feature_engineering.py"]
