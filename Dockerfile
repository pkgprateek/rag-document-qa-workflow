FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency files
COPY requirements.txt .

# Install dependencies with uv (10x faster than pip)
RUN uv pip install --system -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY data/ ./data/

# Expose Gradio default port
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Run the application
CMD ["python", "app/main.py"]
