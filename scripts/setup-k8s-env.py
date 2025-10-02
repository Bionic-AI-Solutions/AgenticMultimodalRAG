#!/usr/bin/env python3
"""
Kubernetes Environment Setup Script
This script helps set up the environment configuration for the Agentic Multimodal RAG System
to connect to the Kubernetes cluster services.
"""

import os
import base64
import subprocess
import json
from pathlib import Path

def run_kubectl_command(command):
    """Run a kubectl command and return the output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"Error running command: {command}")
            print(f"Error: {result.stderr}")
            return None
    except Exception as e:
        print(f"Exception running command: {command}")
        print(f"Exception: {e}")
        return None

def get_k8s_secret(namespace, secret_name, key):
    """Get a secret value from Kubernetes."""
    command = f"kubectl get secret {secret_name} -n {namespace} -o jsonpath='{{.data.{key}}}'"
    encoded_value = run_kubectl_command(command)
    if encoded_value:
        try:
            return base64.b64decode(encoded_value).decode('utf-8')
        except Exception as e:
            print(f"Error decoding secret {secret_name}.{key}: {e}")
            return None
    return None

def get_k8s_service_info(namespace, service_name):
    """Get service information from Kubernetes."""
    command = f"kubectl get service {service_name} -n {namespace} -o json"
    output = run_kubectl_command(command)
    if output:
        try:
            service_info = json.loads(output)
            return {
                'cluster_ip': service_info['spec']['clusterIP'],
                'ports': service_info['spec']['ports']
            }
        except Exception as e:
            print(f"Error parsing service info for {service_name}: {e}")
            return None
    return None

def generate_env_file():
    """Generate the .env file with current Kubernetes cluster information."""
    
    print("🔍 Discovering Kubernetes services...")
    
    # Get service information
    services = {
        'milvus': get_k8s_service_info('milvus', 'milvus'),
        'minio': get_k8s_service_info('minio', 'minio'),
        'postgres': get_k8s_service_info('pg', 'pg-rw'),
        'neo4j': get_k8s_service_info('neo4j', 'neo4j'),
        'redis': get_k8s_service_info('redis-new', 'redis-simple')
    }
    
    # Get secrets
    secrets = {
        'minio_access_key': get_k8s_secret('milvus', 'milvus-minio-secret', 'access-key'),
        'minio_secret_key': get_k8s_secret('milvus', 'milvus-minio-secret', 'secret-key'),
        'neo4j_password': get_k8s_secret('neo4j', 'neo4j', 'password'),
        'postgres_password': get_k8s_secret('pg', 'pg-credentials', 'password'),
        'redis_password': get_k8s_secret('redis-new', 'redis-secrets', 'password')
    }
    
    # Generate environment file content
    env_content = f"""# =============================================================================
# AGENTIC MULTIMODAL RAG SYSTEM - KUBERNETES ENVIRONMENT CONFIGURATION
# =============================================================================
# Generated automatically from Kubernetes cluster
# Generated on: {subprocess.run('date', shell=True, capture_output=True, text=True).stdout.strip()}

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================
NODE_ENV=development
APP_NAME=agentic-multimodal-rag
APP_VERSION=1.0.0
APP_PORT=8000
APP_HOST=0.0.0.0
DEBUG=true
LOG_LEVEL=INFO

# =============================================================================
# MILVUS VECTOR DATABASE CONFIGURATION
# =============================================================================
MILVUS_HOST=milvus.milvus.svc.cluster.local
MILVUS_PORT=19530

# =============================================================================
# MINIO OBJECT STORAGE CONFIGURATION
# =============================================================================
MINIO_HOST=minio.minio.svc.cluster.local
MINIO_PORT=80
MINIO_ACCESS_KEY={secrets.get('minio_access_key', 'x77at4PB02HDuMNXNwr2')}
MINIO_SECRET_KEY={secrets.get('minio_secret_key', 'LY0MkJ1Vawto8K8X4ICqlJ3Drm5I5AzezkWPLztE')}
MINIO_SECURE=false
MINIO_BUCKET=rag-docs
MINIO_REGION=us-east-1

# =============================================================================
# POSTGRESQL DATABASE CONFIGURATION
# =============================================================================
POSTGRES_HOST=pg-rw.pg.svc.cluster.local
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD={secrets.get('postgres_password', 'dfnks.irfheaei;vnc.nvdfighnsnfncxvisruhn')}
POSTGRES_DB=postgres
POSTGRES_SSL=false
POSTGRES_POOL_MIN=2
POSTGRES_POOL_MAX=10
POSTGRES_CONNECTION_TIMEOUT=60000

# Database URL (alternative format)
DATABASE_URL=postgresql://postgres:{secrets.get('postgres_password', 'dfnks.irfheaei;vnc.nvdfighnsnfncxvisruhn')}@pg-rw.pg.svc.cluster.local:5432/postgres

# =============================================================================
# REDIS CONFIGURATION
# =============================================================================
REDIS_HOST=redis-simple.redis-new.svc.cluster.local
REDIS_PORT=6379
REDIS_PASSWORD={secrets.get('redis_password', 'fedfina_staging_redis_password_secure')}
REDIS_DB=0
REDIS_URL=redis://:{secrets.get('redis_password', 'fedfina_staging_redis_password_secure')}@redis-simple.redis-new.svc.cluster.local:6379/0

# Redis Connection Pool
REDIS_POOL_SIZE=10
REDIS_CONNECT_TIMEOUT=10000
REDIS_COMMAND_TIMEOUT=5000

# =============================================================================
# NEO4J GRAPH DATABASE CONFIGURATION
# =============================================================================
NEO4J_URI=bolt://neo4j.neo4j.svc.cluster.local:7687
NEO4J_AUTH=neo4j/{secrets.get('neo4j_password', 'dCqNHU1sgz99lF7h')}
NEO4J_USER=neo4j
NEO4J_PASSWORD={secrets.get('neo4j_password', 'dCqNHU1sgz99lF7h')}

# Neo4j HTTP endpoint (for browser access)
NEO4J_HTTP_URI=http://neo4j.neo4j.svc.cluster.local:7474

# =============================================================================
# AI MODELS CONFIGURATION
# =============================================================================
HF_HOME=/opt/ai-models
TRANSFORMERS_CACHE=/opt/ai-models
HUGGINGFACE_HUB_TOKEN=your-huggingface-token-here
MODEL_DIR=/opt/ai-models

# Jina Embeddings Model
JINA_MODEL_NAME=jinaai/jina-embeddings-v2-base-en
JINA_MODEL_PATH=/opt/ai-models/jinaai__jina-embeddings-v2-base-en

# Nomic Multimodal Model
NOMIC_MODEL_NAME=nomic-ai/colnomic-embed-multimodal-7b
NOMIC_MODEL_PATH=/opt/ai-models/nomic-ai__colnomic-embed-multimodal-7b

# Whisper Model
WHISPER_MODEL_NAME=openai/whisper-base
WHISPER_MODEL_PATH=/opt/ai-models/openai__whisper-base

# =============================================================================
# AUTHENTICATION & SECURITY
# =============================================================================
JWT_SECRET=your-super-secret-jwt-key-min-32-chars-for-rag-system
JWT_EXPIRES_IN=7d
JWT_REFRESH_EXPIRES_IN=30d

# Encryption keys
ENCRYPTION_KEY=your-32-char-encryption-key-here
HASH_ROUNDS=12

# CORS Configuration
CORS_ORIGIN=http://localhost:3000,http://localhost:3001
CORS_CREDENTIALS=true

# =============================================================================
# LOGGING & MONITORING
# =============================================================================
LOG_LEVEL=info
LOG_FORMAT=json
LOG_FILE_ENABLED=true
LOG_FILE_PATH=./logs/app.log
LOG_MAX_FILES=5
LOG_MAX_SIZE=10m

# =============================================================================
# RATE LIMITING
# =============================================================================
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_SKIP_SUCCESSFUL_REQUESTS=false

# =============================================================================
# DEVELOPMENT CONFIGURATION
# =============================================================================
HOT_RELOAD=true
API_DOCS_ENABLED=true
API_DOCS_PATH=/docs

# Testing
TEST_DB_NAME=rag_test_db
TEST_REDIS_DB=1

# =============================================================================
# DEPLOYMENT CONFIGURATION
# =============================================================================
CONTAINER_PORT=8000
HEALTH_CHECK_ENDPOINT=/health
READINESS_CHECK_ENDPOINT=/ready

# Kubernetes
K8S_NAMESPACE=default
K8S_SERVICE_NAME=rag-service

# =============================================================================
# FEATURE FLAGS
# =============================================================================
FEATURE_AGENTIC_REASONING=true
FEATURE_TEMPORAL_KNOWLEDGE=true
FEATURE_VECTOR_SEARCH=true
FEATURE_MULTIMODAL=true
FEATURE_GRAPH_EXPANSION=true
FEATURE_EDGE_GRAPH=true

# =============================================================================
# EDGE GRAPH CONFIGURATION
# =============================================================================
EDGE_GRAPH_CONFIG_PATH=config/edge_graph.yaml
EDGE_GRAPH_HOT_RELOAD=true

# =============================================================================
# FILE PROCESSING CONFIGURATION
# =============================================================================
MAX_FILE_SIZE=104857600  # 100MB
ALLOWED_FILE_TYPES=pdf,txt,docx,jpg,png,mp3,mp4,csv
TEMP_DIR=/tmp/rag-uploads

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================
EMBEDDING_DIMENSION=1024
CHUNK_SIZE=512
CHUNK_OVERLAP=102
SEARCH_TOP_K=10
SEARCH_METRIC_TYPE=IP  # Inner Product

# =============================================================================
# CACHING CONFIGURATION
# =============================================================================
CACHE_TTL=3600  # 1 hour
CACHE_MAX_SIZE=1000
CACHE_ENABLED=true

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================
WORKER_PROCESSES=4
WORKER_CONNECTIONS=1000
KEEPALIVE_TIMEOUT=65
CLIENT_MAX_BODY_SIZE=100M

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================
ALLOWED_HOSTS=localhost,127.0.0.1,192.168.0.0/16
SECURE_SSL_REDIRECT=false
SECURE_HSTS_SECONDS=0
SECURE_HSTS_INCLUDE_SUBDOMAINS=false
SECURE_HSTS_PRELOAD=false
"""
    
    # Write to .env file
    env_file_path = Path('.env')
    with open(env_file_path, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Environment file generated: {env_file_path.absolute()}")
    
    # Print service discovery summary
    print("\n📋 Service Discovery Summary:")
    print("=" * 50)
    
    for service_name, service_info in services.items():
        if service_info:
            print(f"✅ {service_name.upper()}: {service_info['cluster_ip']}")
        else:
            print(f"❌ {service_name.upper()}: Not found")
    
    print("\n🔐 Secrets Retrieved:")
    print("=" * 50)
    
    for secret_name, secret_value in secrets.items():
        if secret_value:
            print(f"✅ {secret_name}: Retrieved")
        else:
            print(f"❌ {secret_name}: Not found")
    
    print(f"\n🚀 Next steps:")
    print(f"1. Review the generated .env file")
    print(f"2. Update any placeholder values (JWT secrets, API keys, etc.)")
    print(f"3. Test connectivity using: python -m app.main")
    print(f"4. Check health endpoints: curl http://localhost:8000/health")

def main():
    """Main function."""
    print("🚀 Kubernetes Environment Setup for Agentic Multimodal RAG")
    print("=" * 60)
    
    # Check if kubectl is available
    kubectl_check = run_kubectl_command("kubectl version --client")
    if not kubectl_check:
        print("❌ kubectl not found. Please install kubectl and configure it to access your cluster.")
        return
    
    print("✅ kubectl found and configured")
    
    # Check cluster connectivity
    cluster_info = run_kubectl_command("kubectl cluster-info")
    if not cluster_info:
        print("❌ Cannot connect to Kubernetes cluster. Please check your kubeconfig.")
        return
    
    print("✅ Connected to Kubernetes cluster")
    
    # Generate environment file
    generate_env_file()

if __name__ == "__main__":
    main()
