from pathlib import Path


# Paths

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
NSL_DIR = BASE_DIR / "data" / "nsl_kdd"

MODEL_DIR = BASE_DIR / "models"

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

KNOWLEDGE_DIR = BASE_DIR / "knowledge"

for directory in (MODEL_DIR, FIGURES_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# The 20 hand-picked CICIDS2017 features of the first version of the project.
# The deployed model uses all features; these are kept as a baseline in the
# feature selection comparison.

HAND_PICKED_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "SYN Flag Count",
    "ACK Flag Count",
    "Average Packet Size"
]


# Columns of the cleaned CICIDS2017 data that are not model inputs

LABEL_COLUMNS = ["Label", "Category", "Target"]


# CICIDS2017 attack categories

NORMAL = "Normal"

CATEGORIES = [
    NORMAL,
    "DoS",
    "PortScan",
    "Brute Force",
    "Botnet",
    "Web Attack",
    "Other"
]

CICIDS_LABEL_MAP = {
    "BENIGN": NORMAL,
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DoS",
    "PortScan": "PortScan",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Bot": "Botnet",
    "Infiltration": "Other",
    "Heartbleed": "Other"
}


def cicids_category(label):

    # Web attack labels contain a broken dash character in the raw CSVs
    if label.startswith("Web Attack"):
        return "Web Attack"

    return CICIDS_LABEL_MAP[label]


# NSL-KDD

NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

NSL_CATEGORICAL = ["protocol_type", "service", "flag"]

NSL_CATEGORIES = [NORMAL, "DoS", "Probe", "R2L", "U2R"]

NSL_ATTACKS = {
    "DoS": [
        "back", "land", "neptune", "pod", "smurf", "teardrop",
        "apache2", "udpstorm", "processtable", "worm", "mailbomb"
    ],
    "Probe": [
        "satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"
    ],
    "R2L": [
        "guess_passwd", "ftp_write", "imap", "phf", "multihop",
        "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
        "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named"
    ],
    "U2R": [
        "buffer_overflow", "loadmodule", "rootkit", "perl",
        "sqlattack", "xterm", "ps"
    ]
}

NSL_LABEL_MAP = {
    attack: category
    for category, attacks in NSL_ATTACKS.items()
    for attack in attacks
}

NSL_LABEL_MAP["normal"] = NORMAL
