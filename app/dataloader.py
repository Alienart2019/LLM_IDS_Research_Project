from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Raw

from app.schemas import LogEvent
from app.config import MAX_PCAP_PACKETS


LABEL_CANDIDATES = [
    "label", "labels", "class", "target", "attack", "attack_type",
    "category", "malicious", "outcome", "y"
]

TEXT_CANDIDATES = [
    "message", "log", "log_text", "text", "event", "payload", "content", "summary"
]

IP_CANDIDATES = ["source_ip", "src_ip", "ip", "src", "source"]
HOST_CANDIDATES = ["hostname", "host", "device", "machine", "sensor"]
SERVICE_CANDIDATES = ["service", "protocol", "app", "application"]


def _normalize_label(value) -> str | None:
    if pd.isna(value):
        return None

    val = str(value).strip().lower()

    benign_values = {"0", "normal", "benign", "false", "no", "clean"}
    suspicious_values = {"suspicious", "anomaly", "anomalous", "unknown", "suspect"}
    malicious_values = {
        "1", "attack", "malicious", "true", "yes", "wiretap", "dos",
        "ddos", "mitm", "mirai", "fuzzing", "scan", "flood",
        "renegotiation", "injection", "exploit", "malware"
    }

    if val in benign_values:
        return "benign"
    if val in suspicious_values:
        return "suspicious"
    if val in malicious_values:
        return "malicious"

    if "benign" in val or "normal" in val:
        return "benign"
    if "suspicious" in val or "anomaly" in val:
        return "suspicious"
    if any(word in val for word in [
        "attack", "malicious", "wiretap", "dos", "ddos",
        "mitm", "mirai", "scan", "flood", "injection", "exploit"
    ]):
        return "malicious"

    return None


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
    return None


def _safe_str_series(series: pd.Series, default: str = "unknown") -> pd.Series:
    return series.fillna(default).astype(str).replace("", default)


def normalize_training_chunk(df: pd.DataFrame) -> pd.DataFrame:
    label_col = _find_column(df, LABEL_CANDIDATES)
    if label_col is None:
        raise ValueError(
            f"Could not find a label column. Expected one of: {LABEL_CANDIDATES}. "
            f"Found columns: {list(df.columns)}"
        )

    text_col = _find_column(df, TEXT_CANDIDATES)
    source_ip_col = _find_column(df, IP_CANDIDATES)
    host_col = _find_column(df, HOST_CANDIDATES)
    service_col = _find_column(df, SERVICE_CANDIDATES)

    working = df.copy()
    working["label"] = working[label_col].apply(_normalize_label)
    working = working.dropna(subset=["label"])

    if working.empty:
        return pd.DataFrame(columns=["text", "label", "source_ip", "hostname", "service"])

    if text_col:
        working["text"] = working[text_col].astype(str)
    else:
        feature_cols = [c for c in working.columns if c != label_col]
        working["text"] = working[feature_cols].astype(str).agg(
            lambda row: " ".join(f"{col}={row[col]}" for col in feature_cols),
            axis=1
        )

    if source_ip_col:
        working["source_ip"] = _safe_str_series(working[source_ip_col])
    else:
        working["source_ip"] = "unknown"

    if host_col:
        working["hostname"] = _safe_str_series(working[host_col])
    else:
        working["hostname"] = "unknown"

    if service_col:
        working["service"] = _safe_str_series(working[service_col])
    else:
        working["service"] = "unknown"

    return working[["text", "label", "source_ip", "hostname", "service"]]


def iter_csv_training_chunks(csv_path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        yield normalize_training_chunk(chunk)


def iter_paired_csv_training_chunks(
    dataset_file: Path,
    labels_file: Path,
    chunk_size: int
) -> Iterable[pd.DataFrame]:
    dataset_iter = pd.read_csv(dataset_file, chunksize=chunk_size, low_memory=False)
    labels_iter = pd.read_csv(labels_file, chunksize=chunk_size, low_memory=False)

    for dataset_chunk, labels_chunk in zip(dataset_iter, labels_iter):
        label_col = _find_column(labels_chunk, LABEL_CANDIDATES)
        if label_col is None:
            if len(labels_chunk.columns) == 1:
                label_col = labels_chunk.columns[0]
            else:
                raise ValueError(
                    f"Could not determine label column from labels file: {list(labels_chunk.columns)}"
                )

        if len(dataset_chunk) != len(labels_chunk):
            raise ValueError(
                f"Chunk row count mismatch between {dataset_file.name} and {labels_file.name}"
            )

        combined = dataset_chunk.copy()
        combined["label"] = labels_chunk[label_col].apply(_normalize_label)
        combined = combined.dropna(subset=["label"])

        if combined.empty:
            continue

        yield normalize_training_chunk(combined)


def packet_to_event(packet) -> LogEvent | None:
    try:
        timestamp = float(getattr(packet, "time", time.time()))
    except Exception:
        timestamp = time.time()

    source_ip = "unknown"
    hostname = "pcap_host"
    service = "unknown"
    parts = []

    if IP in packet:
        source_ip = packet[IP].src
        dst_ip = packet[IP].dst
        proto = packet[IP].proto
        parts.append(f"dst_ip={dst_ip}")
        parts.append(f"ip_proto={proto}")

    if TCP in packet:
        service = "tcp"
        parts.append(f"src_port={packet[TCP].sport}")
        parts.append(f"dst_port={packet[TCP].dport}")
        parts.append(f"tcp_flags={packet[TCP].flags}")
    elif UDP in packet:
        service = "udp"
        parts.append(f"src_port={packet[UDP].sport}")
        parts.append(f"dst_port={packet[UDP].dport}")
    elif ICMP in packet:
        service = "icmp"
        parts.append("icmp=true")

    if Raw in packet:
        try:
            raw_text = bytes(packet[Raw].load)[:200].decode("utf-8", errors="ignore")
            if raw_text.strip():
                parts.append(f"payload={raw_text}")
        except Exception:
            pass

    if not parts:
        return None

    return LogEvent(
        timestamp=timestamp,
        source_ip=source_ip,
        hostname=hostname,
        service=service,
        message=" ".join(parts)
    )


def load_pcap_events(pcap_path: Path) -> list[LogEvent]:
    packets = rdpcap(str(pcap_path), count=MAX_PCAP_PACKETS)
    events: list[LogEvent] = []

    for packet in packets:
        event = packet_to_event(packet)
        if event is not None:
            events.append(event)

    return events


def load_text_log_events(log_path: Path) -> list[LogEvent]:
    events: list[LogEvent] = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            events.append(
                LogEvent(
                    timestamp=time.time(),
                    source_ip="unknown",
                    hostname="log_host",
                    service="log",
                    message=line
                )
            )

    return events


def iter_training_sources(root_path: str):
    root = Path(root_path)

    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root_path}")

    processed_dataset_files: set[Path] = set()

    for dataset_file in root.rglob("*_dataset.csv"):
        base_name = dataset_file.name.replace("_dataset.csv", "")
        folder = dataset_file.parent

        possible_label_files = [
            folder / f"{base_name}_labels.csv",
            folder / f"{base_name.lower()}_labels.csv",
            folder / f"{base_name.upper()}_labels.csv",
        ]

        labels_file = None
        for candidate in possible_label_files:
            if candidate.exists():
                labels_file = candidate
                break

        if labels_file is None:
            label_matches = list(folder.glob("*_labels.csv"))
            if len(label_matches) == 1:
                labels_file = label_matches[0]

        if labels_file is not None:
            processed_dataset_files.add(dataset_file)
            yield "paired_csv", (dataset_file, labels_file)

    for csv_file in root.rglob("*.csv"):
        if csv_file in processed_dataset_files:
            continue
        if csv_file.name.endswith("_labels.csv"):
            continue
        yield "single_csv", csv_file