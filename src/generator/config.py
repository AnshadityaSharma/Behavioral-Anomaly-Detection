from datetime import datetime, timedelta

SEED = 42

N_USERS = 250
N_SERVICE_ACCOUNTS = 40
N_EDGE_DEVICES = 80

SIM_DAYS = 45
SIM_START = datetime(2026, 5, 1)
SIM_END = SIM_START + timedelta(days=SIM_DAYS)

# fraction of total normal sessions each attack type contributes, roughly
# 0.5-3% of sessions per the brief. insider_drift is an ambiguous edge case,
# not a hard "anomaly" ground truth -- kept separate for FP-rate tuning.
ATTACK_RATES = {
    "brute_force": 0.006,
    "impossible_travel": 0.004,
    "credential_stuffing": 0.005,
    "lateral_movement": 0.004,
    "device_spoofing": 0.003,
    "insider_drift": 0.006,
    "low_and_slow_exfil": 0.004,
}

ANOMALY_TYPES = list(ATTACK_RATES.keys())
ALL_LABELS = ["normal"] + ANOMALY_TYPES

AUTH_METHODS = ["password", "token", "certificate", "biometric"]

USER_RESOURCES = [
    "hr/payroll", "finance/ledger", "shared-drive/marketing", "shared-drive/eng",
    "crm/accounts", "crm/leads", "wiki/engineering", "wiki/hr", "vpn-gateway",
    "email/inbox", "billing-portal", "reports/quarterly", "reports/sales",
    "code-repo/frontend", "code-repo/backend", "admin/user-management",
    "admin/access-control", "file-server/legal", "file-server/design",
]

SERVICE_RESOURCES = [
    "svc-auth:8443", "svc-billing:5432", "svc-orders:443", "svc-inventory:8080",
    "svc-notifications:5672", "svc-search:9200", "svc-cache:6379",
    "svc-logging:9000", "svc-metrics:9090", "svc-gateway:443",
]

EDGE_RESOURCES = [
    "telemetry-upload", "firmware-check", "sensor-read", "actuator-control",
    "config-sync", "heartbeat", "diagnostics-report", "ota-update-check",
]

RESOURCES_BY_TYPE = {
    "user": USER_RESOURCES,
    "service_account": SERVICE_RESOURCES,
    "edge_device": EDGE_RESOURCES,
}

PRIVILEGED_ACTIONS = [
    "list_dir", "read_file", "write_file", "delete_file", "run_query",
    "export_data", "create_user", "modify_permissions", "restart_service",
    "read_config", "write_config", "escalate_privilege",
]

DEVICE_OS = [
    "Ubuntu20.04", "Ubuntu22.04", "Windows10", "Windows11", "macOS14",
    "EdgeOS-3.2", "EdgeOS-4.0", "RTOS-1.8",
]
