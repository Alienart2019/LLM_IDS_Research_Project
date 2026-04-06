DB_PATH = "alerts.db"
MODEL_PATH = "models/ids_model.pkl"

USE_LLM = False
LLM_API_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3"

ALLOWLIST_IPS = {"127.0.0.1"}
ALLOWLIST_KEYWORDS = {"trusted_update", "backup_complete"}

DEDUP_WINDOW_SECONDS = 120

SUPPORTED_DATASET_EXTENSIONS = {".csv", ".pcap", ".pcapng", ".log", ".txt"}
MAX_PCAP_PACKETS = 100000