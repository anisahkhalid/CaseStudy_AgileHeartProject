import os
import csv
from datetime import datetime

LOG_PATH = os.path.join("logs", "monitoring_logs.csv")

FIELDNAMES = [
    "timestamp",
    "app_version",
    "model_version",
    "latency_ms",
    "prediction",
    "probability",
    "feedback_score",
    "feedback_comment",
]

def ensure_log_file():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

def log_prediction(app_version, model_version, latency_ms, prediction, probability, feedback_score, feedback_comment):
    ensure_log_file()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "app_version": app_version,
        "model_version": model_version,
        "latency_ms": round(float(latency_ms), 2),
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "feedback_score": int(feedback_score),
        "feedback_comment": (feedback_comment or "").strip(),
    }
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
