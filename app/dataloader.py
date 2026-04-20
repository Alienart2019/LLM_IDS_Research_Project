"""
dataloader.py — unified dataset loading for training and runtime detection.

Responsibilities
----------------
1. **Training sources.** Walk a data directory and yield every ingestible
   CSV, handling both the paired ``*_dataset.csv`` + ``*_labels.csv`` layout
   used by the Kitsune benchmark and single-file labeled CSVs.
2. **Label normalization.** Map the wide variety of raw label values
   (``"0"``, ``"1"``, ``"Mirai"``, ``"benign"``, ``"normal"``, ``True``, etc.)
   into the three classes the model knows: ``benign``, ``suspicious``,
   ``malicious``.
3. **Runtime event loading.** Parse ``.pcap`` / ``.pcapng`` captures and
   plain-text log files into :class:`LogEvent` objects for live detection.

Everything is chunk-based where possible so the pipeline stays memory-safe
on multi-gigabyte datasets like Kitsune.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Raw

from app.schemas import LogEvent
from app.config import MAX_PCAP_PACKETS


# ---------------------------------------------------------------------------
# Column-name candidates
# ---------------------------------------------------------------------------
# When we look at an unfamiliar CSV, we don't know what the columns are
# called. These lists are checked (case-insensitive) to map a real column to
# a conceptual role. The first match wins.

LABEL_CANDIDATES = [
    "label", "labels", "class", "target", "attack", "attack_type",
    "category", "malicious", "outcome", "y",
    # Kitsune's labels file uses a bare "x" column for the 0/1 flag.
    "x",
]

TEXT_CANDIDATES = [
    "message", "log", "log_text", "text", "event", "payload", "content", "summary"
]

IP_CANDIDATES = ["source_ip", "src_ip", "ip", "src", "source"]
HOST_CANDIDATES = ["hostname", "host", "device", "machine", "sensor"]
SERVICE_CANDIDATES = ["service", "protocol", "app", "application"]


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------


def _normalize_label(value) -> str | None:
    """
    Coerce an arbitrary label value into ``benign`` / ``suspicious`` /
    ``malicious``.

    Returns ``None`` if the value can't be interpreted — the caller is
    expected to drop such rows before training.
    """
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

    # Fall through to substring matching for less common label variants.
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
    """Return the first column in ``df`` that matches any of ``candidates``
    (case-insensitive). Returns ``None`` if no candidate is present."""
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
    return None


def _safe_str_series(series: pd.Series, default: str = "unknown") -> pd.Series:
    """Cast a Series to string, replacing NaN and empty strings with ``default``."""
    return series.fillna(default).astype(str).replace("", default)


# ---------------------------------------------------------------------------
# Training chunk normalization
# ---------------------------------------------------------------------------


def normalize_training_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw training chunk into the canonical training schema.

    The output has exactly these columns:

    ======================  ===============================================
    text                    Flat text representation the vectorizer sees.
    label                   One of benign / suspicious / malicious.
    source_ip, hostname,    Best-effort context used by feature extraction.
    service
    ======================  ===============================================

    Rows whose label cannot be normalized are dropped.

    Raises
    ------
    ValueError
        If no label column can be found in the input DataFrame.
    """
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
        # No natural text column — serialize every feature as "key=value"
        # pairs. This is what feeds the hashing vectorizer downstream.
        feature_cols = [c for c in working.columns if c != label_col]
        working["text"] = working[feature_cols].astype(str).agg(
            lambda row: " ".join(f"{col}={row[col]}" for col in feature_cols),
            axis=1
        )

    working["source_ip"] = (
        _safe_str_series(working[source_ip_col]) if source_ip_col else "unknown"
    )
    working["hostname"] = (
        _safe_str_series(working[host_col]) if host_col else "unknown"
    )
    working["service"] = (
        _safe_str_series(working[service_col]) if service_col else "unknown"
    )

    return working[["text", "label", "source_ip", "hostname", "service"]]


def iter_csv_training_chunks(csv_path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    """Yield normalized chunks from a single labeled CSV."""
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        yield normalize_training_chunk(chunk)


def iter_paired_csv_training_chunks(
    dataset_file: Path,
    labels_file: Path,
    chunk_size: int
) -> Iterable[pd.DataFrame]:
    """
    Yield normalized chunks from a paired features/labels CSV layout.

    The two files are read in lock-step so row ``i`` of the features chunk
    is paired with row ``i`` of the labels chunk.

    The Kitsune labels files use a layout like::

        ,x
        0,0
        1,1

    Both the ``Unnamed: 0`` index and the ``x`` label column confuse naive
    "single column == the label" heuristics. This loader handles all three
    cases:

    1. An explicitly-named label column from :data:`LABEL_CANDIDATES`.
    2. A one-column labels file (the column is assumed to be the label).
    3. A two-column labels file whose first column is an index and whose
       second column is the label (the Kitsune case).
    """
    dataset_iter = pd.read_csv(dataset_file, chunksize=chunk_size, low_memory=False)
    labels_iter = pd.read_csv(labels_file, chunksize=chunk_size, low_memory=False)

    for dataset_chunk, labels_chunk in zip(dataset_iter, labels_iter):
        label_col = _find_column(labels_chunk, LABEL_CANDIDATES)

        if label_col is None:
            # Case 2: single-column labels file.
            if len(labels_chunk.columns) == 1:
                label_col = labels_chunk.columns[0]
            else:
                # Case 3: strip obvious index columns (Unnamed: 0) and see
                # if exactly one usable column remains.
                non_index = [
                    c for c in labels_chunk.columns
                    if not str(c).lower().startswith("unnamed")
                ]
                if len(non_index) == 1:
                    label_col = non_index[0]
                else:
                    raise ValueError(
                        f"Could not determine label column from labels file: "
                        f"{list(labels_chunk.columns)}"
                    )

        if len(dataset_chunk) != len(labels_chunk):
            raise ValueError(
                f"Chunk row count mismatch between {dataset_file.name} "
                f"and {labels_file.name}"
            )

        combined = dataset_chunk.copy()
        combined["label"] = labels_chunk[label_col].apply(_normalize_label)
        combined = combined.dropna(subset=["label"])

        if combined.empty:
            continue

        yield normalize_training_chunk(combined)


# ---------------------------------------------------------------------------
# Runtime event loaders (PCAP and text logs)
# ---------------------------------------------------------------------------


def packet_to_event(packet) -> LogEvent | None:
    """
    Convert a single scapy packet into a :class:`LogEvent`.

    Returns ``None`` for packets we can't meaningfully describe (e.g. link-
    layer-only frames without IP, TCP, UDP, ICMP, or a usable payload).

    The resulting ``message`` field is a flat string of ``key=value``
    tokens — the same format the training text representation uses — so
    the model sees PCAP-derived events in a shape it recognizes.
    """
    try:
        timestamp = float(getattr(packet, "time", time.time()))
    except Exception:
        timestamp = time.time()

    source_ip = "unknown"
    hostname = "pcap_host"
    service = "unknown"
    parts: list[str] = []

    if IP in packet:
        source_ip = packet[IP].src
        parts.append(f"dst_ip={packet[IP].dst}")
        parts.append(f"ip_proto={packet[IP].proto}")

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
        message=" ".join(parts),
    )


def load_pcap_events(pcap_path: Path) -> list[LogEvent]:
    """Parse a PCAP into :class:`LogEvent` objects, capped at ``MAX_PCAP_PACKETS``."""
    packets = rdpcap(str(pcap_path), count=MAX_PCAP_PACKETS)
    events: list[LogEvent] = []
    for packet in packets:
        event = packet_to_event(packet)
        if event is not None:
            events.append(event)
    return events


def load_text_log_events(log_path: Path) -> list[LogEvent]:
    """Parse a plain-text log file into one :class:`LogEvent` per non-empty line."""
    events: list[LogEvent] = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(
                LogEvent(
                    timestamp=time.time(),
                    source_ip="unknown",
                    hostname="log_host",
                    service="log",
                    message=line,
                )
            )
    return events


# ---------------------------------------------------------------------------
# Training source discovery
# ---------------------------------------------------------------------------


def iter_training_sources(root_path: str):
    """
    Walk a data directory and yield training sources in priority order.

    Each yielded value is a ``(kind, payload)`` tuple where ``kind`` is
    either ``"paired_csv"`` (payload is ``(dataset_file, labels_file)``)
    or ``"single_csv"`` (payload is a single path). Paired CSVs are yielded
    first and their matching ``_dataset.csv`` is excluded from the single-
    CSV pass so it isn't double-processed.
    """
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
