"""
dataset_config.py
------------------
Fallback dataset used only when the user does not upload a file.
The connection string now comes from the environment (.env) instead of
being hard-coded, so the password never lives in source control.

  DEFAULT_DB_URI=postgresql+psycopg2://user:pass@localhost:5432/sales_db
  DEFAULT_DB_TABLE=orders

If DEFAULT_DB_URI is empty, DATASET_CONFIG is None and the website simply
hides the "use the default database" button.
"""
from config import DEFAULT_DB_TABLE, DEFAULT_DB_URI

DATASET_CONFIG = (
    {
        "type": "sql",
        "connection_uri": DEFAULT_DB_URI,
        "target": DEFAULT_DB_TABLE,
        "description": "Default configured database table.",
    }
    if DEFAULT_DB_URI
    else None
)
