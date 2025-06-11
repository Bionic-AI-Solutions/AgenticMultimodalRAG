from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="jinaai/jina-embeddings-v2-base-en",
    local_dir="/Volumes/ssd/mac/models/jinaai__jina-embeddings-v2-base-en-test",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=3
)
print("Download complete. Check /Volumes/ssd/mac/models/jinaai__jina-embeddings-v2-base-en-test for files.") 