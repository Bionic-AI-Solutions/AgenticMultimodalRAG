import os
import logging
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Configurable model directory
MODEL_DIR = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/Volumes/ssd/models"
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_models")

# Model specs
MODELS = [
    ("jinaai/jina-embeddings-v2-base-en", "sentence-transformers"),
    ("BAAI/bge-m3", "sentence-transformers"),
    ("nomic-ai/colnomic-embed-multimodal-7b", "huggingface"),
    ("openai/whisper-base", "huggingface"),
]

def download_sentence_transformer(model_name):
    logger.info(f"Downloading SentenceTransformer model: {model_name}")
    SentenceTransformer(model_name, cache_folder=MODEL_DIR, trust_remote_code=True)

def download_huggingface(model_name):
    logger.info(f"Downloading HuggingFace model: {model_name}")
    snapshot_download(repo_id=model_name, local_dir=os.path.join(MODEL_DIR, model_name), local_dir_use_symlinks=False)

def main():
    for model_name, model_type in MODELS:
        try:
            if model_type == "sentence-transformers":
                download_sentence_transformer(model_name)
            elif model_type == "huggingface":
                download_huggingface(model_name)
            else:
                logger.warning(f"Unknown model type: {model_type} for {model_name}")
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
    logger.info("All models processed.")

if __name__ == "__main__":
    main() 