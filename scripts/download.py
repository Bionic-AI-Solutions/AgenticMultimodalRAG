import os
import logging
import hashlib
import concurrent.futures
import time
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm
import sys
import json
from typing import Dict, List, Optional, Any, Tuple
import requests
from pathlib import Path
import platform
import torch
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.file_hash_manager import update_hash, get_stored_hash, verify_or_download, compute_sha256

# Load .env if present
load_dotenv()

# Configurable model directory
MODEL_DIR = os.getenv("MODEL_DIR") or os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/Volumes/ssd/mac/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("download_models")
# Set file_hash_manager logger to DEBUG for detailed hash logs
logging.getLogger("file_hash_manager").setLevel(logging.DEBUG)

# Define device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model specs with detailed metadata
MODELS = [
    {
        "name": "jinaai/jina-embeddings-v2-base-en",
        "type": "sentence-transformers",
        "version": "v2.0.0",
        "description": "Primary text embedding model",
        "metadata": {
            "dimensions": 768,
            "max_sequence_length": 8192,
            "language": "en",
            "architecture": "BERT",
            "license": "apache-2.0",
            "size": "137M parameters",
            "performance_metrics": {
                "speed": "~1000 sentences/second on CPU",
                "memory_usage": "~500MB",
                "accuracy": "0.85 on STS-B",
            },
            "use_cases": ["Text similarity", "Semantic search", "Document clustering"],
            "requirements": {"min_memory": "4GB", "recommended_memory": "8GB", "gpu_optional": True},
            "checksums": {},  # Will be populated dynamically
        },
    },
    {
        "name": "BAAI/bge-m3",
        "type": "sentence-transformers",
        "version": "v1.0.0",
        "description": "Fallback text embedding model",
        "metadata": {
            "dimensions": 1024,
            "max_sequence_length": 8192,
            "language": "multilingual",
            "architecture": "XLM-RoBERTa",
            "license": "mit",
            "size": "1.1B parameters",
            "performance_metrics": {
                "speed": "~500 sentences/second on CPU",
                "memory_usage": "~2GB",
                "accuracy": "0.88 on STS-B",
            },
            "use_cases": ["Multilingual search", "Cross-lingual similarity", "Document retrieval"],
            "requirements": {"min_memory": "8GB", "recommended_memory": "16GB", "gpu_recommended": True},
            "checksums": {},  # Will be populated dynamically
        },
    },
    {
        "name": "nomic-ai/colnomic-embed-multimodal-7b",
        "type": "huggingface",
        "version": "v1.0.0",
        "description": "Multimodal embedding model for images and PDFs",
        "metadata": {
            "dimensions": 4096,
            "max_sequence_length": 8192,
            "modalities": ["text", "image", "pdf"],
            "architecture": "Qwen2.5",
            "license": "apache-2.0",
            "size": "7B parameters",
            "performance_metrics": {
                "speed": "~100 images/second on GPU",
                "memory_usage": "~14GB",
                "accuracy": "0.92 on image-text matching",
            },
            "use_cases": ["Image search", "Document understanding", "Multimodal retrieval"],
            "requirements": {"min_memory": "16GB", "recommended_memory": "32GB", "gpu_required": True},
            "checksums": {},  # Will be populated dynamically
        },
    },
    {
        "name": "openai/whisper-base",
        "type": "huggingface",
        "version": "v1.0.0",
        "description": "Audio transcription model",
        "metadata": {
            "dimensions": 512,
            "max_sequence_length": 480000,
            "modalities": ["audio"],
            "architecture": "Transformer",
            "license": "mit",
            "size": "74M parameters",
            "performance_metrics": {
                "speed": "~1x real-time on CPU",
                "memory_usage": "~1GB",
                "accuracy": "0.85 WER on LibriSpeech",
            },
            "use_cases": ["Speech recognition", "Audio transcription", "Voice search"],
            "requirements": {"min_memory": "2GB", "recommended_memory": "4GB", "gpu_optional": True},
            "checksums": {},  # Will be populated dynamically
        },
    },
]

# Dynamically set essential files for Nomic multimodal model to all .safetensors, .json, and README.md files in the repo
for model in MODELS:
    if model["name"] == "nomic-ai/colnomic-embed-multimodal-7b":
        from huggingface_hub import HfApi

        api = HfApi()
        repo_files = api.list_repo_files(model["name"])
        model["essential_files"] = [f for f in repo_files if f.endswith((".safetensors", ".json")) or f == "README.md"]
    if model["name"] == "openai/whisper-base":
        # Ensure both .safetensors and .bin are considered essential
        from huggingface_hub import HfApi

        api = HfApi()
        repo_files = api.list_repo_files(model["name"])
        # Only add if present in repo
        whisper_essentials = [f for f in repo_files if f in ("model.safetensors", "pytorch_model.bin")]
        model["essential_files"] = whisper_essentials


def get_system_info() -> Dict[str, Any]:
    """Get system information for metadata"""
    return {
        "python_version": platform.python_version(),
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_info": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        ),
    }


def retry_on_exception(exception_types: Tuple[Exception, ...]):
    """Create a retry condition for specific exception types"""
    return retry_if_exception_type(exception_types)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_on_exception((requests.exceptions.RequestException, OSError)),
)
def fetch_file_with_retry(api: HfApi, model_name: str, file: str) -> Optional[str]:
    """Fetch a single file with retry mechanism and log why download is triggered."""
    try:
        model_dir = (
            os.getenv("MODEL_DIR") or os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/Volumes/ssd/mac/models"
        )
        model_path = os.path.join(model_dir, model_name.replace("/", "__"))
        file_path = os.path.join(model_path, file)
        expected_hash = get_stored_hash(file_path)

        def download_func(path):
            api.hf_hub_download(
                repo_id=model_name,
                filename=file,
                repo_type="model",
                cache_dir=model_path,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )

        verify_or_download(file_path, expected_hash, download_func)
        # After download, update hash
        update_hash(file_path)
        return file_path
    except Exception as e:
        logger.error(f"Failed to fetch file {file} for model {model_name}: {e}")
        return None


def get_actual_model_path(model_info: Dict[str, Any]) -> str:
    """Get the actual path where the model is downloaded"""
    model_name = model_info["name"]
    # Use double underscore convention for all models
    model_path = os.path.join(MODEL_DIR, model_name.replace("/", "__"))
    return model_path


def verify_file_checksum(file_path: str, expected_hash: str) -> Tuple[str, bool, Optional[str]]:
    """Verify checksum for a single file"""
    try:
        if not os.path.exists(file_path):
            return file_path, False, "File not found"

        actual_hash = compute_sha256(file_path)
        is_valid = actual_hash == expected_hash

        if not is_valid:
            return file_path, False, f"Expected: {expected_hash}, Actual: {actual_hash}"

        return file_path, True, None
    except Exception as e:
        return file_path, False, str(e)


def verify_model_checksums(model_info: Dict[str, Any], model_dir: str) -> bool:
    """Verify checksums for a downloaded model with parallel processing"""
    if not model_info["metadata"]["checksums"]:
        logger.warning(f"No checksums available for {model_info['name']}")
        return True

    # Get the actual model path
    try:
        actual_model_path = get_actual_model_path(model_info)
        logger.info(f"Actual model path for {model_info['name']}: {actual_model_path}")
    except Exception as e:
        logger.error(f"Failed to get actual model path for {model_info['name']}: {str(e)}")
        return False

    if not os.path.exists(actual_model_path):
        logger.error(f"Model directory not found: {actual_model_path}")
        return False

    # Prepare files for verification
    files_to_verify = []
    for file in os.listdir(actual_model_path):
        if file.endswith((".bin", ".safetensors", ".json", ".model", ".txt")):
            if file in model_info["metadata"]["checksums"]:
                file_path = os.path.join(actual_model_path, file)
                expected_hash = model_info["metadata"]["checksums"][file]
                files_to_verify.append((file_path, expected_hash))

    if not files_to_verify:
        logger.warning(f"No files to verify for {model_info['name']}")
        return True

    # Determine optimal number of workers
    system_info = get_system_info()
    max_workers = min(8, len(files_to_verify), (system_info["gpu_count"] + 1) * 2)

    all_valid = True
    failed_files = []

    with tqdm(total=len(files_to_verify), desc=f"Verifying {model_info['name']}", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all verification tasks
            future_to_file = {
                executor.submit(verify_file_checksum, file_path, expected_hash): file_path
                for file_path, expected_hash in files_to_verify
            }

            # Process results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    _, is_valid, error_msg = future.result()
                    if not is_valid:
                        all_valid = False
                        failed_files.append((file_path, error_msg))
                        logger.error(f"Checksum verification failed for {file_path}: {error_msg}")
                except Exception as e:
                    all_valid = False
                    failed_files.append((file_path, str(e)))
                    logger.error(f"Error verifying {file_path}: {str(e)}")
                finally:
                    pbar.update(1)

    # Report summary
    if failed_files:
        logger.error(f"\nChecksum verification failed for {len(failed_files)} files in {model_info['name']}:")
        for file_path, error_msg in failed_files:
            logger.error(f"- {os.path.basename(file_path)}: {error_msg}")
    else:
        logger.info(f"All {len(files_to_verify)} files verified successfully for {model_info['name']}")

    return all_valid


def fetch_model_checksums(model_name: str) -> Dict[str, str]:
    """Fetch actual checksums from HuggingFace with parallel processing"""
    try:
        api = HfApi()
        files = api.list_repo_files(model_name)
        checksums = {}

        # Filter relevant files
        relevant_files = [f for f in files if f.endswith((".bin", ".safetensors", ".json", ".model", ".txt"))]

        # Determine optimal number of workers
        system_info = get_system_info()
        max_workers = min(4, len(relevant_files), (system_info["gpu_count"] + 1))

        with tqdm(total=len(relevant_files), desc=f"Fetching checksums for {model_name}", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all fetch tasks
                future_to_file = {
                    executor.submit(fetch_file_with_retry, api, model_name, file): file for file in relevant_files
                }

                # Process results as they complete
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        file_info = future.result()
                        if file_info and os.path.exists(file_info):
                            checksums[file] = compute_sha256(file_info)
                            # Clean up the downloaded file
                            os.remove(file_info)
                    except Exception as e:
                        logger.error(f"Failed to get checksum for {file} after retries: {str(e)}")
                    finally:
                        pbar.update(1)

        return checksums
    except Exception as e:
        logger.error(f"Failed to fetch checksums for {model_name}: {str(e)}")
        return {}


def calculate_file_hash(file_path: str, hash_type: str = "sha256") -> str:
    """Calculate hash of a file"""
    hash_func = getattr(hashlib, hash_type)()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return f"{hash_type}:{hash_func.hexdigest()}"


def verify_model_download(model_info: Dict) -> bool:
    """Verify that the model was downloaded correctly"""
    model_name = model_info["name"]
    # Always use double-underscore convention for all models
    model_path = os.path.join(MODEL_DIR, model_name.replace("/", "__"))

    if not os.path.exists(model_path):
        return False

    # Check for essential files
    essential_files = ["config.json"]
    if model_info["type"] == "sentence-transformers":
        essential_files.extend(["modules.json", "sentence_bert_config.json"])

    # Verify all essential files exist and have correct checksums
    for file_name in essential_files:
        file_path = os.path.join(model_path, file_name)
        if not os.path.exists(file_path):
            return False
        if file_name in model_info["metadata"]["checksums"]:
            if not verify_file_checksum(file_path, model_info["metadata"]["checksums"][file_name]):
                return False

    return True


def save_model_metadata(model_info: Dict, success: bool):
    """Save model metadata and download status"""
    model_name = model_info["name"]
    metadata_path = os.path.join(MODEL_DIR, f"{model_name.replace('/', '__')}_metadata.json")

    metadata = {
        "name": model_name,
        "type": model_info["type"],
        "version": model_info["version"],
        "description": model_info["description"],
        "metadata": model_info["metadata"],
        "download_status": "success" if success else "failed",
        "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": os.path.join(MODEL_DIR, model_name.replace("/", "__")),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def get_essential_files(model_info: Dict[str, Any], repo_files: List[str]) -> List[str]:
    """Determine the essential files for a model, either from model_info or by inferring from repo."""
    if "essential_files" in model_info:
        return model_info["essential_files"]
    # Default: all .json, .bin, .safetensors, .txt files in repo
    return [f for f in repo_files if f.endswith((".json", ".bin", ".safetensors", ".txt"))]


def get_abs_path(model_dir, rel_path):
    """Return the absolute path for a file given the model directory and its relative path."""
    return os.path.abspath(os.path.join(model_dir, rel_path))


def download_model(model_info: Dict[str, Any]) -> bool:
    """Download all essential files for a model, always verifying and updating hashes."""
    from huggingface_hub import HfApi

    model_name = model_info["name"]
    model_dir = os.path.join(MODEL_DIR, model_name.replace("/", "__"))
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Checking {model_name} in {model_dir}")

    api = HfApi()
    repo_files = api.list_repo_files(model_name)
    essential_files = get_essential_files(model_info, repo_files)

    for file_name in essential_files:
        file_path = get_abs_path(model_dir, file_name)
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        logger.debug(f"Checking file: {file_path} (exists: {os.path.exists(file_path)})")
        expected_hash = model_info["metadata"].get("checksums", {}).get(file_name)
        needs_download = False
        if not os.path.exists(file_path):
            logger.info(f"File missing: {file_name}")
            needs_download = True
        elif expected_hash:
            actual_hash = compute_sha256(file_path)
            if actual_hash != expected_hash:
                logger.info(f"Hash mismatch for {file_name}: expected {expected_hash}, got {actual_hash}")
                needs_download = True
        if needs_download:
            try:
                hf_hub_download(
                    repo_id=model_name,
                    filename=file_name,
                    cache_dir=model_dir,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                logger.info(f"Downloaded {file_name} for {model_name}")
            except Exception as e:
                logger.error(f"Failed to download {file_name} for {model_name}: {e}")
                return False
        # Always compute and store hash after download
        if os.path.exists(file_path):
            update_hash(file_path)

    # Final verification: all essential files must exist and match hash if available
    for file_name in essential_files:
        file_path = get_abs_path(model_dir, file_name)
        if not os.path.exists(file_path):
            logger.error(f"File still missing after download: {file_name}")
            return False
        expected_hash = model_info["metadata"].get("checksums", {}).get(file_name)
        if expected_hash:
            actual_hash = compute_sha256(file_path)
            if actual_hash != expected_hash:
                logger.error(f"Hash mismatch after download for {file_name}: expected {expected_hash}, got {actual_hash}")
                return False
    logger.info(f"All essential files for {model_name} are now present and valid.")
    return True


def main():
    total_models = len(MODELS)
    successful_downloads = 0
    summary = []  # To store status for each model

    logger.info(f"Starting download of {total_models} models to {MODEL_DIR}")

    # Fetch checksums for all models first
    logger.info("Fetching model checksums...")
    for model_info in tqdm(MODELS, desc="Fetching checksums", unit="model"):
        model_info["metadata"]["checksums"] = fetch_model_checksums(model_info["name"])

    # Adjust max_workers based on system capabilities
    system_info = get_system_info()
    max_workers = min(3, system_info["gpu_count"] + 1)  # Use GPU count + 1, but max 3

    logger.info(f"Using {max_workers} parallel downloads based on system capabilities")

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_model = {executor.submit(download_model, model_info): model_info for model_info in MODELS}

        # Process results as they complete
        for future in tqdm(as_completed(future_to_model), total=len(MODELS), desc="Downloading models"):
            model_info = future_to_model[future]
            try:
                result = future.result()
                if result:
                    successful_downloads += 1
                    summary.append({"model": model_info["name"], "status": "SUCCESS", "reason": ""})
                else:
                    summary.append(
                        {
                            "model": model_info["name"],
                            "status": "FAILED",
                            "reason": "Verification or download failed. See logs.",
                        }
                    )
            except Exception as e:
                summary.append({"model": model_info["name"], "status": "FAILED", "reason": str(e)})

    logger.info(f"Download complete. Successfully downloaded {successful_downloads}/{total_models} models")

    if successful_downloads < total_models:
        logger.warning("Some models failed to download. Check the logs for details.")

    # Print summary table
    print("\nMODEL DOWNLOAD SUMMARY:")
    print("{:<40} {:<10} {}".format("Model", "Status", "Reason"))
    print("-" * 80)
    for entry in summary:
        print("{:<40} {:<10} {}".format(entry["model"], entry["status"], entry["reason"]))

    if successful_downloads < total_models:
        sys.exit(1)


if __name__ == "__main__":
    main()
