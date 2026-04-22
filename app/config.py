"""
config.py — All runtime configuration for the Trainable IDS.

Every knob is sourced from an environment variable with a safe default.
Import individual names directly::

    from app.config import MODEL_PATH, USE_LLM

"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DB_PATH             = os.getenv("IDS_DB_PATH", "alerts.db")
MODEL_PATH          = os.getenv("IDS_MODEL_PATH", "models/ids_model.pkl")
HYBRID_MODEL_PATH   = os.getenv("IDS_HYBRID_MODEL_PATH", "models/ids_hybrid_model.pkl")

# ---------------------------------------------------------------------------
# LLM explainer
# ---------------------------------------------------------------------------
USE_LLM     = os.getenv("IDS_USE_LLM", "false").lower() == "true"
LLM_API_URL = os.getenv("IDS_LLM_API_URL", "http://localhost:11434/api/generate")
LLM_MODEL   = os.getenv("IDS_LLM_MODEL", "llama3")

# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------
_allowlist_ips_raw = os.getenv("IDS_ALLOWLIST_IPS", "127.0.0.1")
ALLOWLIST_IPS: set[str] = set(
    ip.strip() for ip in _allowlist_ips_raw.split(",") if ip.strip()
)
_allowlist_kw_raw = os.getenv("IDS_ALLOWLIST_KEYWORDS", "trusted_update,backup_complete")
ALLOWLIST_KEYWORDS: set[str] = set(
    kw.strip() for kw in _allowlist_kw_raw.split(",") if kw.strip()
)

# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
DEDUP_WINDOW_SECONDS = int(os.getenv("IDS_DEDUP_WINDOW_SECONDS", "120"))

# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
SUPPORTED_DATASET_EXTENSIONS = {".csv", ".pcap", ".pcapng", ".log", ".txt"}
MAX_PCAP_PACKETS = int(os.getenv("IDS_MAX_PCAP_PACKETS", "100000"))

# ---------------------------------------------------------------------------
# Honeypot
# ---------------------------------------------------------------------------
HONEYPOT_ENABLED  = os.getenv("IDS_HONEYPOT_ENABLED", "false").lower() == "true"
HONEYPOT_HOST     = os.getenv("IDS_HONEYPOT_HOST", "honeypot")
HONEYPOT_PORT     = int(os.getenv("IDS_HONEYPOT_PORT", "8080"))
HONEYPOT_URL      = os.getenv(
    "IDS_HONEYPOT_URL",
    f"http://{HONEYPOT_HOST}:{HONEYPOT_PORT}/incoming"
)
_honeypot_labels_raw = os.getenv("IDS_HONEYPOT_LABELS", "malicious")
HONEYPOT_TRIGGER_LABELS: set[str] = set(
    lbl.strip() for lbl in _honeypot_labels_raw.split(",") if lbl.strip()
)
HONEYPOT_FORWARD_TIMEOUT = float(os.getenv("IDS_HONEYPOT_FORWARD_TIMEOUT", "5.0"))

# ---------------------------------------------------------------------------
# API security
# ---------------------------------------------------------------------------
API_KEY        = os.getenv("IDS_API_KEY", "")
API_RATE_LIMIT = int(os.getenv("IDS_API_RATE_LIMIT", "200"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("IDS_LOG_LEVEL", "INFO").upper()
LOG_JSON  = os.getenv("IDS_LOG_JSON", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Hybrid model — Phase 1
# ---------------------------------------------------------------------------
USE_HYBRID_DETECTOR = os.getenv("IDS_USE_HYBRID", "false").lower() == "true"

# ---------------------------------------------------------------------------
# SMOTE — training only
# ---------------------------------------------------------------------------
USE_SMOTE = os.getenv("IDS_USE_SMOTE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# IoT-specific rules — Phase 1
# ---------------------------------------------------------------------------
IOT_RULES_ENABLED       = os.getenv("IDS_IOT_RULES", "false").lower() == "true"
IOT_DOS_RATE_THRESHOLD  = int(os.getenv("IDS_IOT_DOS_RATE", "100"))
IOT_BLACKHOLE_SIGMA     = float(os.getenv("IDS_IOT_BLACKHOLE_SIGMA", "3.0"))

# ---------------------------------------------------------------------------
# Deployment tier
# ---------------------------------------------------------------------------
DEPLOYMENT_TIER = os.getenv("IDS_DEPLOYMENT_TIER", "cloud").lower()

# ---------------------------------------------------------------------------
# Physical-Cyber Security Fusion — Phase 2
# ---------------------------------------------------------------------------
PHYSICAL_FUSION_ENABLED         = os.getenv("IDS_PHYSICAL_FUSION", "false").lower() == "true"
PIR_GATEWAY_URL                 = os.getenv("IDS_PIR_GATEWAY_URL", "http://pir-gateway:5000/pir/state")
PIR_POLL_INTERVAL_SECONDS       = float(os.getenv("IDS_PIR_POLL_INTERVAL", "2.0"))
PIR_MOTION_WINDOW_SECONDS       = float(os.getenv("IDS_PIR_MOTION_WINDOW", "30.0"))
CAMERA_GATEWAY_URL              = os.getenv("IDS_CAMERA_GATEWAY_URL", "http://camera-gw:8090")
CAMERA_RECORD_COOLDOWN_SECONDS  = float(os.getenv("IDS_CAMERA_COOLDOWN", "300.0"))
GSM_GATEWAY_URL                 = os.getenv("IDS_GSM_GATEWAY_URL", "")
GSM_ALERT_PHONE                 = os.getenv("IDS_GSM_ALERT_PHONE", "")
_fusion_labels_raw = os.getenv("IDS_PHYSICAL_FUSION_LABELS", "malicious,suspicious")
PHYSICAL_FUSION_TRIGGER_LABELS: set[str] = set(
    lbl.strip() for lbl in _fusion_labels_raw.split(",") if lbl.strip()
)

# ---------------------------------------------------------------------------
# Behavioral Profiler (AMoF + DS Hierarchy) — Phase 2
# ---------------------------------------------------------------------------
BEHAVIORAL_PROFILER_ENABLED     = os.getenv("IDS_BEHAVIORAL_PROFILER", "false").lower() == "true"
AMOF_TICK_INTERVAL_SECONDS      = float(os.getenv("IDS_AMOF_TICK_INTERVAL", "30.0"))
AMOF_FLUCTUATION_THRESHOLD      = float(os.getenv("IDS_AMOF_THRESHOLD", "5.0"))
CCI_CONFIDENCE_THRESHOLD        = float(os.getenv("IDS_CCI_THRESHOLD", "0.75"))
SUPER_NODE_CONSENSUS_FRACTION   = float(os.getenv("IDS_SUPER_CONSENSUS", "0.6"))

# ---------------------------------------------------------------------------
# Complex Event Processing (CEP) — Phase 2
# ---------------------------------------------------------------------------
CEP_ENABLED           = os.getenv("IDS_CEP_ENABLED", "false").lower() == "true"
CEP_PATTERN_OVERRIDES = os.getenv("IDS_CEP_PATTERNS", "")

# ---------------------------------------------------------------------------
# Trust Management (NBTD / CNTD) — Phase 2
# ---------------------------------------------------------------------------
TRUST_MANAGER_ENABLED = os.getenv("IDS_TRUST_ENABLED", "false").lower() == "true"
TRUST_STRATEGY        = os.getenv("IDS_TRUST_STRATEGY", "nbtd").lower()

# ---------------------------------------------------------------------------
# Gateway Monitor (Packet Drop Probability / SPRT) — Phase 2
# ---------------------------------------------------------------------------
GATEWAY_MONITOR_ENABLED = os.getenv("IDS_GATEWAY_MONITOR", "false").lower() == "true"
GATEWAY_PDP_NORMAL      = float(os.getenv("IDS_GW_PDP_NORMAL", "0.01"))
GATEWAY_PDP_ATTACK      = float(os.getenv("IDS_GW_PDP_ATTACK", "0.15"))

# ---------------------------------------------------------------------------
# Artificial Immune System (AIS) — Phase 2
# ---------------------------------------------------------------------------
AIS_ENABLED            = os.getenv("IDS_AIS_ENABLED", "false").lower() == "true"
AIS_AFFINITY_THRESHOLD = float(os.getenv("IDS_AIS_AFFINITY", "0.65"))
AIS_CLONE_THRESHOLD    = int(os.getenv("IDS_AIS_CLONE_THRESHOLD", "3"))
AIS_MAX_MEMORY         = int(os.getenv("IDS_AIS_MAX_MEMORY", "10000"))
