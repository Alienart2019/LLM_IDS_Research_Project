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
    message = str(row.get("message", ""))
    service = str(row.get("service", "unknown"))
    hostname = str(row.get("hostname", "unknown"))
    source_ip = str(row.get("source_ip", "unknown"))

    msg = message.lower()
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

    return (
        f"service={service} "
        f"host={hostname} "
        f"ip={source_ip} "
        f"message={message} "
        f"flags={' '.join(flags)}"
    )
