import os

# --- Paths ---
DB_PATH = os.getenv("IDS_DB_PATH", "alerts.db")
MODEL_PATH = os.getenv("IDS_MODEL_PATH", "models/ids_model.pkl")
HYBRID_MODEL_PATH = os.getenv("IDS_HYBRID_MODEL_PATH", "models/ids_hybrid_model.pkl")

# --- LLM ---
USE_LLM = os.getenv("IDS_USE_LLM", "false").lower() == "true"
LLM_API_URL = os.getenv("IDS_LLM_API_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("IDS_LLM_MODEL", "llama3")

# --- Allowlists ---
_allowlist_ips_raw = os.getenv("IDS_ALLOWLIST_IPS", "127.0.0.1")
ALLOWLIST_IPS: set[str] = set(ip.strip() for ip in _allowlist_ips_raw.split(",") if ip.strip())

_allowlist_kw_raw = os.getenv("IDS_ALLOWLIST_KEYWORDS", "trusted_update,backup_complete")
ALLOWLIST_KEYWORDS: set[str] = set(kw.strip() for kw in _allowlist_kw_raw.split(",") if kw.strip())

# --- Dedup ---
DEDUP_WINDOW_SECONDS = int(os.getenv("IDS_DEDUP_WINDOW_SECONDS", "120"))

# --- Dataset loader ---
SUPPORTED_DATASET_EXTENSIONS = {".csv", ".pcap", ".pcapng", ".log", ".txt"}
MAX_PCAP_PACKETS = int(os.getenv("IDS_MAX_PCAP_PACKETS", "100000"))

# --- Honeypot ---
HONEYPOT_ENABLED = os.getenv("IDS_HONEYPOT_ENABLED", "false").lower() == "true"
HONEYPOT_HOST = os.getenv("IDS_HONEYPOT_HOST", "honeypot")
HONEYPOT_PORT = int(os.getenv("IDS_HONEYPOT_PORT", "8080"))
HONEYPOT_URL = os.getenv(
    "IDS_HONEYPOT_URL",
    f"http://{HONEYPOT_HOST}:{HONEYPOT_PORT}/incoming"
)
_honeypot_labels_raw = os.getenv("IDS_HONEYPOT_LABELS", "malicious")
HONEYPOT_TRIGGER_LABELS: set[str] = set(
    lbl.strip() for lbl in _honeypot_labels_raw.split(",") if lbl.strip()
)
HONEYPOT_FORWARD_TIMEOUT = float(os.getenv("IDS_HONEYPOT_FORWARD_TIMEOUT", "5.0"))

# --- API security ---
API_KEY = os.getenv("IDS_API_KEY", "")
API_RATE_LIMIT = int(os.getenv("IDS_API_RATE_LIMIT", "200"))

# --- Logging ---
LOG_LEVEL = os.getenv("IDS_LOG_LEVEL", "INFO").upper()
LOG_JSON = os.getenv("IDS_LOG_JSON", "false").lower() == "true"

# --- Hybrid model ---
# Set IDS_USE_HYBRID=true to use the 2-stage K-means + DT detector alongside
# (or instead of) the primary SGD model.
USE_HYBRID_DETECTOR = os.getenv("IDS_USE_HYBRID", "false").lower() == "true"

# --- SMOTE ---
# Set IDS_USE_SMOTE=true during training (train_hybrid.py --smote) to enable
# Synthetic Minority Over-sampling.  Has no effect at inference time.
USE_SMOTE = os.getenv("IDS_USE_SMOTE", "false").lower() == "true"

# --- IoT-specific rules ---
# Enables the stateful IoT rule engine (wormhole, black-hole, DoS flood).
IOT_RULES_ENABLED = os.getenv("IDS_IOT_RULES", "false").lower() == "true"

# DoS flood threshold (packets/second per source IP).
IOT_DOS_RATE_THRESHOLD = int(os.getenv("IDS_IOT_DOS_RATE", "100"))

# Black-hole AMoF deviation threshold (sigma).
IOT_BLACKHOLE_SIGMA = float(os.getenv("IDS_IOT_BLACKHOLE_SIGMA", "3.0"))

# --- Deployment tier ---
# "edge"  — lightweight models only (XGBoost/RandomForest or the DT from the
#            hybrid, targeting sub-2 MB on-disk size).
# "cloud" — full pipeline including hybrid deep models (default).
DEPLOYMENT_TIER = os.getenv("IDS_DEPLOYMENT_TIER", "cloud").lower()
