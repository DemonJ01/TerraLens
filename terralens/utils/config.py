from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a dict."""
    p = Path(path)
    with p.open("r") as f:
        return yaml.safe_load(f)
