FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install python-dotenv

EXPOSE 8501

CMD python3 -m streamlit run streamlit/app.py --server.port=8501

