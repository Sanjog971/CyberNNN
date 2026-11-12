import csv
from pathlib import Path

DATA_FILE = Path("data") / "raw" / "sample_connections.csv"


def load_rows(path):
    """Read a CSV file and return a list of dictionaries, one per row."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        return []


def count_labels(rows):
    """Return a dictionary like {'BENIGN': 3, 'ATTACK': 3}."""
    counts = {}
    for row in rows:
        label = row["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def total_bytes(rows):
    """Add up bytes_sent, skipping values that are not valid integers."""
    total = 0
    for row in rows:
        try:
            total += int(row["bytes_sent"])
        except ValueError:
            print(f"Skipping bad value: {row['bytes_sent']!r}")
    return total


rows = load_rows(DATA_FILE)

def ssh_attempts(rows):
    """Count rows whose destination port is 22 (SSH)."""
    count = 0
    for row in rows:
        if row["dst_port"] == "22":
            count += 1
    return count

print("SSH attempts:", ssh_attempts(rows))
print("Rows loaded:", len(rows))
print("Label counts:", count_labels(rows))
print("Total bytes sent:", total_bytes(rows))