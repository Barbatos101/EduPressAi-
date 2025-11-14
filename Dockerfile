FROM python:3.10-slim

WORKDIR /code

# Install system dependencies including Hindi language support
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-script-deva \
    libgl1 \
    libglib2.0-0 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for optimization
ENV HF_HOME=/tmp/.huggingface
ENV TRANSFORMERS_CACHE=/tmp/.huggingface
ENV HUGGINGFACE_HUB_CACHE=/tmp/.huggingface
ENV HOME=/tmp
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 user
RUN chown -R user /code /tmp
USER user

# Expose port
EXPOSE 7860

# Run Streamlit with optimized settings
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.maxUploadSize=15", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

