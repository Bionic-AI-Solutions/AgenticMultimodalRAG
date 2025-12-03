# DevContainer Setup Summary

## Files Created

1. **`.devcontainer/devcontainer.json`** - Main devcontainer configuration
   - Builds from custom Dockerfile
   - Configures VS Code extensions and settings
   - Sets up environment variables
   - Configures port forwarding

2. **`.devcontainer/Dockerfile`** - Custom development container image
   - Python 3.11 base
   - Miniconda with `test` environment
   - Poetry for dependency management
   - Docker CLI for docker-compose
   - All required system dependencies

3. **`.devcontainer/post-create.sh`** - Post-creation setup script
   - Installs Poetry dependencies
   - Creates .env file if missing
   - Starts services via docker-compose
   - Checks service health

4. **`.devcontainer/README.md`** - Comprehensive documentation
   - Usage instructions
   - Common tasks
   - Troubleshooting guide

5. **`.devcontainer/docker-compose.yml`** - Placeholder (services use main docker-compose.yml)

## Key Features

✅ **Python 3.11** via Conda environment (`test`)  
✅ **Poetry** for dependency management  
✅ **Docker & Docker Compose** for services  
✅ **VS Code Extensions** pre-installed (Python, Docker, Git, etc.)  
✅ **Persistent volumes** for models and caches  
✅ **Cross-platform** support (Linux, macOS, Windows)  
✅ **Auto-activation** of conda environment  

## Quick Start

1. Open project in VS Code
2. Press `F1` → "Dev Containers: Reopen in Container"
3. Wait for container to build and setup
4. Run `docker-compose up -d` to start services
5. Start coding!

## Service Connection

The devcontainer connects to services using `host.docker.internal` for cross-platform compatibility:
- On Linux: Uses host gateway
- On macOS/Windows: Uses Docker's special hostname

Services should be started from the host or within the container using:
```bash
docker-compose up -d
```

## Environment Variables

All necessary environment variables are set in `devcontainer.json`:
- `HF_HOME=/opt/ai-models`
- `TRANSFORMERS_CACHE=/opt/ai-models`
- `MODEL_DIR=/opt/ai-models`
- `PYTHONPATH=/workspace`
- `ENV=development`

## Volumes

- `/workspace` - Project code (mounted from host)
- `/opt/ai-models` - AI models cache (persisted in `.devcontainer/ai-models/`)
- Poetry cache (Docker volume)
- Conda packages cache (Docker volume)

## Notes

- The devcontainer uses Docker-in-Docker feature to run docker-compose
- Services run on the host network for easy access
- All Python dependencies are managed via Poetry
- The conda environment `test` is automatically activated in new terminals


