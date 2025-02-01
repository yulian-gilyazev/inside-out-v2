# Readme

### App launch
```sh
docker build -t inside-out-v2-streamlit-app:latest .
docker run --env-file .env --name streamlit-inside-out-v2 -p 8501:8501 inside-out-v2-streamlit-app
```