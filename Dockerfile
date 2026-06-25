FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app/chat_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]