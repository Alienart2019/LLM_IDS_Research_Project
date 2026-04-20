"""
features.py — feature engineering helpers.

There are two feature pipelines in this project and both live here:

* :func:`extract_rule_flags` runs a set of regexes over the event message
  and returns a list of named IOCs (failed_login, recon_tool, etc.). These
  flags appear verbatim on every :class:`~app.schemas.DetectionResult`.
* :func:`event_to_text` and :func:`row_to_text` produce the flat text
  representation that the hashing vectorizer consumes. They're used both at
  training time (via :mod:`app.trainer`) and at inference time (via
  :mod:`app.detector`), so that the model sees the same token shape in
  both environments.
"""

import re
from typing import Dict

from app.schemas import LogEvent


def extract_rule_flags(event: LogEvent) -> list[str]:
    """
    Apply a set of regex-based rules to ``event.message`` and return the
    names of every rule that fired.

    The order of the returned list is stable (same order as the rules
    below) so downstream code can use it directly as a dedup key.

    The rules are intentionally coarse-grained — they're meant as a safety
    net of obvious-bad signals, not a full detection ruleset.
    """
    msg = event.message.lower()
    flags: list[str] = []

    # --- Authentication-related -----------------------------------------
    if "failed password" in msg or "login failed" in msg:
        flags.append("failed_login")
    if "invalid user" in msg:
        flags.append("invalid_user")
    if "sudo" in msg and ("not in sudoers" in msg or "authentication failure" in msg):
        flags.append("privilege_escalation")

    # --- Offensive tooling ----------------------------------------------
    if re.search(r"(nmap|masscan|nikto)", msg, re.IGNORECASE):
        flags.append("recon_tool")
    if re.search(r"(wget|curl).*(http|https)://", msg, re.IGNORECASE):
        flags.append("remote_download")
    if re.search(r"(powershell|cmd\.exe|bash -i|nc |whoami|net user|base64)", msg, re.IGNORECASE):
        flags.append("suspicious_command")

    # --- Network-layer indicators (set by pcap/log ingestion) -----------
    if re.search(r"(tcp_flags=s|suspicious_tcp_pattern=true)", msg, re.IGNORECASE):
        flags.append("scan_like_tcp")
    if re.search(r"(dst_port=3389|dst_port=445|dst_port=23)", msg, re.IGNORECASE):
        flags.append("high_risk_port")

    # --- Payload-level indicators ---------------------------------------
    if re.search(r"(payload=.*select.+from|payload=.*union.+select|payload=.*drop table)",
                 msg, re.IGNORECASE):
        flags.append("sql_injection_like")

    return flags


def event_to_text(event: LogEvent) -> str:
    """
    Flatten a :class:`LogEvent` into the exact string shape the vectorizer
    was trained on.

    The format is a series of ``key=value`` tokens terminated by a space-
    separated ``flags=`` field. Keeping this function in sync with
    :func:`row_to_text` is what makes training and inference interchangeable.
    """
    flags = extract_rule_flags(event)
    return (
        f"service={event.service} "
        f"host={event.hostname} "
        f"ip={event.source_ip} "
        f"message={event.message} "
        f"flags={' '.join(flags)}"
    )


def row_to_text(row: Dict) -> str:
    """
    Convert a single training DataFrame row (as a dict) into the same flat
    text form :func:`event_to_text` produces.

    Missing fields fall back to ``"unknown"`` so training on partially
    populated rows still works. This function is called via
    ``DataFrame.apply`` inside :mod:`app.trainer`.
    """
    message = str(row.get("message", row.get("text", "")))
    service = str(row.get("service", "unknown"))
    hostname = str(row.get("hostname", "unknown"))
    source_ip = str(row.get("source_ip", "unknown"))

    event = LogEvent(
        timestamp=0.0,
        source_ip=source_ip,
        hostname=hostname,
        service=service,
        message=message,
    )
    return event_to_text(event)
