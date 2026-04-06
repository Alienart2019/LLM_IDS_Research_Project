import re
from typing import Dict
from app.schemas import LogEvent


def extract_rule_flags(event: LogEvent) -> list[str]:
    msg = event.message.lower()
    flags = []

    if "failed password" in msg or "login failed" in msg:
        flags.append("failed_login")

    if "invalid user" in msg:
        flags.append("invalid_user")

    if "sudo" in msg and ("not in sudoers" in msg or "authentication failure" in msg):
        flags.append("privilege_escalation")

    if re.search(r"(nmap|masscan|nikto)", msg, re.IGNORECASE):
        flags.append("recon_tool")

    if re.search(r"(wget|curl).*(http|https)://", msg, re.IGNORECASE):
        flags.append("remote_download")

    if re.search(r"(powershell|cmd\.exe|bash -i|nc |whoami|net user|base64)", msg, re.IGNORECASE):
        flags.append("suspicious_command")

    if re.search(r"(tcp_flags=s|suspicious_tcp_pattern=true)", msg, re.IGNORECASE):
        flags.append("scan_like_tcp")

    if re.search(r"(dst_port=3389|dst_port=445|dst_port=23)", msg, re.IGNORECASE):
        flags.append("high_risk_port")

    if re.search(r"(payload=.*select.+from|payload=.*union.+select|payload=.*drop table)", msg, re.IGNORECASE):
        flags.append("sql_injection_like")

    return flags


def event_to_text(event: LogEvent) -> str:
    flags = extract_rule_flags(event)
    return (
        f"service={event.service} "
        f"host={event.hostname} "
        f"ip={event.source_ip} "
        f"message={event.message} "
        f"flags={' '.join(flags)}"
    )


def row_to_text(row: Dict) -> str:
    message = str(row.get("message", row.get("text", "")))
    service = str(row.get("service", "unknown"))
    hostname = str(row.get("hostname", "unknown"))
    source_ip = str(row.get("source_ip", "unknown"))

    event = LogEvent(
        timestamp=0.0,
        source_ip=source_ip,
        hostname=hostname,
        service=service,
        message=message
    )
    return event_to_text(event)