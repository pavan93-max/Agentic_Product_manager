import json
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)
MEMORY_FILE = Path("memory/experiment_memory.json")


def _ensure_memory_file() -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        initial_data = {"experiments": []}
        with open(MEMORY_FILE, "w") as f:
            json.dump(initial_data, f, indent=2)


def log_experiment(record: Dict[str, Any]) -> None:
    _ensure_memory_file()
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        if "experiments" not in data:
            data["experiments"] = []
        data["experiments"].append(record)
        temp_file = MEMORY_FILE.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(MEMORY_FILE)
        logger.info(f"Logged experiment with decision: {record.get('decision', 'UNKNOWN')}")
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to log experiment: {e}")
        raise


def get_experiments() -> List[Dict[str, Any]]:
    _ensure_memory_file()
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        return data.get("experiments", [])
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read experiments: {e}")
        return []
