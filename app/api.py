from fastapi import FastAPI
from app.storage import get_alerts

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Trainable IDS API running"}


@app.get("/alerts")
def alerts(limit: int = 50):
    return get_alerts(limit)
