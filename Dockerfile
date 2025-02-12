
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV OPENAI_API_KEY=$OPENAI_API_KEY
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
