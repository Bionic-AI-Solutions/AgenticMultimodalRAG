import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Callable

HASHES_FILE = Path(__file__).parent / "file_hashes.json"


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hashes() -> dict:
    if not HASHES_FILE.exists():
        return {}
    with open(HASHES_FILE, "r") as f:
        return json.load(f)


def save_hashes(hashes: dict):
    tmp_file = str(HASHES_FILE) + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(hashes, f, indent=2)
    os.replace(tmp_file, HASHES_FILE)


def update_hash(filepath: str):
    hashes = load_hashes()
    file_hash = compute_sha256(filepath)
    hashes[filepath] = file_hash
    save_hashes(hashes)
    return file_hash


def verify_or_download(
    filepath: str,
    expected_hash: Optional[str],
    download_func: Callable[[str], None],
    *download_args,
    **download_kwargs
) -> bool:
    """
    Verifies the file at filepath against the expected_hash. If missing or hash mismatch, calls download_func(filepath, *args, **kwargs), then updates the hash.
    Returns True if file is valid after this call, False otherwise.
    """
    if os.path.exists(filepath):
        current_hash = compute_sha256(filepath)
        if expected_hash is not None and expected_hash == current_hash:
            return True
        else:
            print(f"[file_hash_manager] {filepath} hash mismatch or unknown. Re-downloading.")
    else:
        print(f"[file_hash_manager] {filepath} missing. Downloading.")

    # Download and update hash
    download_func(filepath, *download_args, **download_kwargs)
    new_hash = compute_sha256(filepath)
    hashes = load_hashes()
    hashes[filepath] = new_hash
    save_hashes(hashes)
    if expected_hash is not None and new_hash != expected_hash:
        print(f"[file_hash_manager] Warning: Downloaded file hash still does not match expected.")
        return False
    return True


def get_stored_hash(filepath: str) -> Optional[str]:
    hashes = load_hashes()
    return hashes.get(filepath) 