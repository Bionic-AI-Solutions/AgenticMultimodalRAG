# DevContainer Setup for Agentic Multimodal RAG

This devcontainer provides a complete development environment for the Agentic Multimodal RAG system.

## Prerequisites

- Docker Desktop or Docker Engine with Docker Compose
- Visual Studio Code with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## Quick Start

1. **Open in VS Code**: Open this folder in VS Code
2. **Reopen in Container**: Press `F1` → "Dev Containers: Reopen in Container"
3. **Wait for Setup**: The container will build and install all dependencies
4. **Start Services**: Run `docker-compose up -d` from the workspace root to start all required services

## What's Included

### Development Tools
- **Python 3.11** via Conda environment (`test`)
- **Poetry** for dependency management
- **Docker & Docker Compose** for running services
- **Git** and common development utilities
- **Zsh** with Oh My Zsh for better terminal experience

### VS Code Extensions
- Python language support (Pylance, Black, isort, Flake8)
- Docker support
- GitLens
- GitHub Copilot
- Jupyter notebook support
- YAML and JSON support

### Services (via docker-compose.yml)
- **PostgreSQL** (port 5432)
- **Neo4j** (ports 7474, 7687)
- **Milvus** (port 19530)
- **MinIO** (ports 9000, 9001)
- **Redis** (port 6379)
- **etcd** (port 2379)

## Environment Variables

The devcontainer sets up the following environment variables:
- `PYTHONUNBUFFERED=1`
- `PYTHONDONTWRITEBYTECODE=1`
- `PYTHONPATH=/workspace`
- `HF_HOME=/models` (mounted from host `/mnt/ai-models`)
- `TRANSFORMERS_CACHE=/models` (mounted from host `/mnt/ai-models`)
- `MODEL_DIR=/models` (mounted from host `/mnt/ai-models`)
- `ENV=development`

## Common Tasks

### Starting the Application
```bash
conda activate test
poetry run uvicorn app.main:app --reload
```

### Running Tests
```bash
# Unit tests
ENV=test poetry run pytest tests/unit

# Integration tests
ENV=test poetry run pytest tests/integratione2e

# All tests
ENV=test poetry run pytest
```

### Code Quality
```bash
# Format code
poetry run black .

# Sort imports
poetry run isort .

# Lint code
poetry run flake8 .

# Type checking
poetry run mypy .
```

### Managing Services
```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View service logs
docker-compose logs -f

# Check service status
docker-compose ps
```

### Downloading Models
```bash
# Run the download script (models will be saved to /models, mounted from host /mnt/ai-models)
ENV=development poetry run python scripts/download.py

# Verify models are accessible
ls -la /models
```

## Volumes

The devcontainer uses the following volumes:
- `/workspace` - Your project code (mounted from host)
- `/models` - AI models directory (mounted from host `/mnt/ai-models`)
- `/opt/ai-models` - Fallback AI models cache (persisted in `.devcontainer/ai-models/`)
- Poetry cache (persisted Docker volume)
- Conda packages cache (persisted Docker volume)

**Note**: The primary models directory is `/models` which is mounted from `/mnt/ai-models` on the host. Ensure this directory exists on your host system and contains your AI models.

## Troubleshooting

### Services Not Starting
If services fail to start, check:
1. Docker is running: `docker ps`
2. Ports are not in use: Check if ports 5432, 7474, 7687, 9000, 19530, 6379 are available
3. Check logs: `docker-compose logs`

### Python Environment Issues
```bash
# Reinstall dependencies
conda activate test
poetry install --no-interaction

# Clear Poetry cache
poetry cache clear pypi --all
```

### Model Download Issues
- Ensure you have sufficient disk space in `/opt/ai-models`
- Check network connectivity for HuggingFace downloads
- Verify `.env` file has correct `HF_HOME` and `MODEL_DIR` settings

## Notes

- The devcontainer uses `host` network mode to connect to services
- Services should be started manually using `docker-compose up -d`
- The conda environment `test` is automatically activated in new terminals
- All Python dependencies are managed via Poetry

