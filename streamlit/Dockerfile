FROM harbor.wildberries.ru/docker-hub-proxy/python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install python-dotenv
RUN pip install opencv-python==4.8.0.74
RUN pip install opencv-python-headless==4.8.0.74

COPY . /app

EXPOSE 8501

CMD ["python3", "-m", "streamlit", "run", "streamlit/app.py", "--server.port=8501"]
