"""Keep reviewed source holds through final result-file metadata rewrites.

Historical hashes remain bound. A held canonical source path stays held even
when its final bytes change; this does not claim the new bytes were reviewed.
"""

import hashlib
import json
from pathlib import Path
import re


def held_source_hashes(audit_directory):
    held = set()

    def inspect(value):
        if isinstance(value, dict):
            if value.get("held_for_training") is True:
                digest = value.get("source_sha256")
                assert isinstance(digest, str) and re.fullmatch(
                    r"[0-9a-f]{64}", digest
                ), "Unbound training hold"
                held.add(digest)
                source = value.get("source") or value.get("file")
                if source:
                    assert isinstance(source, str), "Invalid held source path"
                    path = Path(source)
                    assert path.is_absolute() and path.name == "result.json", (
                        "Invalid held result path"
                    )
                    if path.is_file():
                        held.add(hashlib.sha256(path.read_bytes()).hexdigest())
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    for review in sorted(audit_directory.glob("*.json")):
        inspect(json.loads(review.read_text()))
    return held
