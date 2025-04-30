# Readme

Place `.env` file in the root of the project and set `OPENAI_API_KEY` variable.


### App launch with docker compose
```sh
docker compose up -d
```


### App launch with docker
```sh
docker build -t inside-out-v2-streamlit-app:latest .
docker run --env-file .env --name streamlit-inside-out-v2 -p 8501:8501 inside-out-v2-streamlit-app
```
