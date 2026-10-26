# src/features.py
# Shared by the notebooks AND the API, so training and serving use identical preprocessing.
import numpy as np
import pandas as pd

# 41 features + label + difficulty (the raw NSL-KDD files have no header row)
COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_srv_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
FEATURES = [c for c in COLS if c not in ("label", "difficulty")]

CATEGORICAL = ["protocol_type", "service", "flag"]   # text columns -> one-hot
LOG_COLS = ["duration", "src_bytes", "dst_bytes"]    # huge-range columns -> log1p
DROP_COLS = ["num_outbound_cmds"]                    # constant column (see EDA)


def build_matrix(df, columns=None):
    """Raw flow rows -> numeric table (not yet scaled).

    columns=None  : training mode, the column list comes from this data.
    columns=[...] : use the saved column list. Missing columns become 0 and
                    unseen category values are ignored (the API uses this).
    """
    X = df.drop(columns=["label", "difficulty", "is_attack"] + DROP_COLS,
                errors="ignore").copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    X = pd.get_dummies(X, columns=CATEGORICAL, dtype=float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X
