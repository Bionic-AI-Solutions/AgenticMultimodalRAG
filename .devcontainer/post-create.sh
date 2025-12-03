#!/bin/bash
set -e

echo "🚀 Setting up Agentic Multimodal RAG development environment..."

# Ensure directories are owned by vscode user (created by devcontainer features)
# Note: /workspace is mounted from host, so we need to be careful with permissions
if [ -d "/models" ]; then
    sudo chown -R vscode:vscode /models 2>/dev/null || true
fi
if [ -d "/opt/ai-models" ]; then
    sudo chown -R vscode:vscode /opt/ai-models 2>/dev/null || true
fi
# For /workspace, we ensure vscode user can read/write, but preserve host ownership structure
if [ -d "/workspace" ]; then
    # Check if workspace is empty or has files
    if [ "$(ls -A /workspace 2>/dev/null)" ]; then
        echo "✅ Workspace directory mounted and contains files"
        # Ensure vscode user has read/write access without changing ownership
        sudo chmod -R u+rw /workspace 2>/dev/null || true
    else
        echo "⚠️  Workspace directory is empty - checking mount..."
        # If empty, try to set ownership (might be a new mount)
        sudo chown -R vscode:vscode /workspace 2>/dev/null || true
    fi
fi

# Initialize conda for zsh if not already done
if ! grep -q "conda shell.zsh hook" ~/.zshrc 2>/dev/null; then
    echo 'eval "$(/opt/conda/bin/conda shell.zsh hook)"' >> ~/.zshrc
    echo 'conda activate test' >> ~/.zshrc
fi

# Activate conda environment
source /opt/conda/bin/activate test

# Configure Poetry (as vscode user)
poetry config virtualenvs.create false

# Install Poetry dependencies
echo "📦 Installing Poetry dependencies..."
cd /workspace
poetry install --no-interaction --no-root

# Install the project itself
echo "📦 Installing project..."
poetry install --no-interaction

# Create .env file if it doesn't exist
if [ ! -f /workspace/.env ]; then
    echo "📝 Creating .env file from template..."
    if [ -f /workspace/.env.example ]; then
        cp /workspace/.env.example /workspace/.env
    else
        cat > /workspace/.env << EOF
# Application Configuration
ENV=development
LOG_LEVEL=INFO

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=rag_db

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_AUTH=neo4j/neo4jpassword

# MinIO Configuration
MINIO_HOST=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
MINIO_BUCKET=rag-docs

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Model Configuration (mounted from host /mnt/ai-models)
HF_HOME=/models
TRANSFORMERS_CACHE=/models
MODEL_DIR=/models
EOF
    fi
    echo "✅ Created .env file"
fi

# Start services using docker-compose (if docker-compose is available)
# Note: docker-compose might be available via Docker CLI (docker compose) or standalone
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    echo "🐳 Starting services with docker-compose..."
    cd /workspace
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d || echo "⚠️  Could not start services. Make sure docker-compose.yml exists and Docker is running."
    else
        docker compose up -d || echo "⚠️  Could not start services. Make sure docker-compose.yml exists and Docker is running."
    fi
    
    # Wait for services to be ready
    echo "⏳ Waiting for services to be ready..."
    sleep 10
else
    echo "⚠️  docker-compose not found. Services need to be started manually from the host:"
    echo "   cd /home/skadam/git/AgenticMultimodalRAG"
    echo "   docker-compose up -d"
    echo ""
    echo "   Or install docker-compose in the container if Docker socket is mounted"
fi

# Check service health
echo "🔍 Checking service health..."
python3 << 'PYEOF'
import sys
import time
import socket
import os

# Use host.docker.internal for cross-platform compatibility
# On Linux with host network, this might not resolve, so try localhost as fallback
def check_port(host, port, timeout=5):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

# Try host.docker.internal first, then localhost as fallback
def check_service(name, port):
    hosts = ["host.docker.internal", "localhost"]
    for host in hosts:
        if check_port(host, port):
            print(f"✅ {name} is ready (via {host})")
            return True
    print(f"⚠️  {name} is not ready (may need more time or manual start)")
    return False

services = {
    "PostgreSQL": 5432,
    "Neo4j": 7687,
    "Milvus": 19530,
    "MinIO": 9000,
    "Redis": 6379,
}

print("Checking services...")
for name, port in services.items():
    check_service(name, port)
PYEOF

# Check if models directory is mounted
echo ""
if [ -d "/models" ] && [ "$(ls -A /models 2>/dev/null)" ]; then
    echo "✅ Models directory mounted at /models (from host /mnt/ai-models)"
    echo "   Found $(ls -1 /models | wc -l) items in models directory"
elif [ -d "/models" ]; then
    echo "⚠️  Models directory /models exists but is empty"
    echo "   Ensure /mnt/ai-models exists on host and contains model files"
else
    echo "⚠️  Models directory /models not found"
    echo "   Ensure /mnt/ai-models exists on host"
fi

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "📚 Next steps:"
echo "   1. Ensure services are running: docker-compose up -d"
echo "   2. Run 'conda activate test' to activate the conda environment"
echo "   3. Run 'poetry run uvicorn app.main:app --reload' to start the API"
echo "   4. Run 'ENV=test poetry run pytest tests/unit' for unit tests"
echo "   5. Run 'ENV=test poetry run pytest tests/integratione2e' for integration tests"
echo ""
echo "🔧 Useful commands:"
echo "   - Activate conda: conda activate test"
echo "   - Start services: docker-compose up -d"
echo "   - Stop services: docker-compose down"
echo "   - Run tests: ENV=test poetry run pytest"
echo "   - Format code: poetry run black ."
echo "   - Lint code: poetry run flake8 ."
echo "   - Type check: poetry run mypy ."
echo ""

