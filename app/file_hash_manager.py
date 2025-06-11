import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Callable
import logging

HASHES_FILE = Path(__file__).parent / "file_hashes.json"
logger = logging.getLogger("file_hash_manager")


def compute_sha256(filepath: str) -> str:
    abs_path = os.path.abspath(filepath)
    h = hashlib.sha256()
    with open(abs_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    hash_val = h.hexdigest()
    logger.debug(f"[compute_sha256] {abs_path} hash: {hash_val}")
    return hash_val


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
    abs_path = os.path.abspath(filepath)
    hashes = load_hashes()
    file_hash = compute_sha256(abs_path)
    hashes[abs_path] = file_hash
    save_hashes(hashes)
    logger.info(f"[update_hash] Updated hash for {abs_path}: {file_hash}")
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
    abs_path = os.path.abspath(filepath)
    stored_hash = get_stored_hash(abs_path)
    logger.info(f"[verify_or_download] Checking {abs_path}\n  - expected_hash: {expected_hash}\n  - stored_hash: {stored_hash}")
    if os.path.exists(abs_path):
        current_hash = compute_sha256(abs_path)
        logger.info(f"[verify_or_download] {abs_path} exists. current_hash: {current_hash}")
        if expected_hash is not None and expected_hash == current_hash:
            logger.info(f"[verify_or_download] {abs_path} hash matches expected. No download needed.")
            return True
        else:
            logger.warning(f"[verify_or_download] {abs_path} hash mismatch or unknown. Re-downloading.")
    else:
        logger.warning(f"[verify_or_download] {abs_path} missing. Downloading.")

    # Download and update hash
    download_func(abs_path, *download_args, **download_kwargs)
    new_hash = compute_sha256(abs_path)
    hashes = load_hashes()
    hashes[abs_path] = new_hash
    save_hashes(hashes)
    logger.info(f"[verify_or_download] Downloaded and updated hash for {abs_path}: {new_hash}")
    if expected_hash is not None and new_hash != expected_hash:
        logger.warning(f"[verify_or_download] Warning: Downloaded file hash still does not match expected.\n  - expected: {expected_hash}\n  - actual: {new_hash}")
        return False
    return True


def get_stored_hash(filepath: str) -> Optional[str]:
    abs_path = os.path.abspath(filepath)
    hashes = load_hashes()
    hash_val = hashes.get(abs_path)
    logger.debug(f"[get_stored_hash] {abs_path} stored_hash: {hash_val}")
    return hash_val 