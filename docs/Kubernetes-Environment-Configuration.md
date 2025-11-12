# Kubernetes Environment Configuration

This document provides the complete environment configuration for the Agentic Multimodal RAG System to connect to the Kubernetes cluster services.

## Overview

Based on the analysis of the current implementation and the Kubernetes cluster, the following services are available and configured:

## Service Discovery

### 1. Milvus Vector Database
- **Service**: `milvus.milvus.svc.cluster.local:19530`
- **Purpose**: Vector storage for embeddings
- **Configuration**:
  ```bash
  MILVUS_HOST=milvus.milvus.svc.cluster.local
  MILVUS_PORT=19530
  ```

### 2. MinIO Object Storage
- **Service**: `minio.minio.svc.cluster.local:80`
- **External Access**: NodePort `30900` on cluster IPs
- **Purpose**: File storage for uploaded documents
- **Credentials** (from Kubernetes secrets):
  - Access Key: `x77at4PB02HDuMNXNwr2`
  - Secret Key: `LY0MkJ1Vawto8K8X4ICqlJ3Drm5I5AzezkWPLztE`
- **Configuration**:
  ```bash
  MINIO_HOST=minio.minio.svc.cluster.local
  MINIO_PORT=80
  MINIO_ACCESS_KEY=x77at4PB02HDuMNXNwr2
  MINIO_SECRET_KEY=LY0MkJ1Vawto8K8X4ICqlJ3Drm5I5AzezkWPLztE
  MINIO_SECURE=false
  MINIO_BUCKET=rag-docs
  ```

### 3. PostgreSQL Database
- **Service**: `pg-haproxy-primary.pg.svc.cluster.local:5432` (HAProxy LoadBalancer)
- **Purpose**: Relational data storage with high availability via HAProxy
- **Benefits**: Automatic failover, load balancing, and connection pooling
- **Credentials** (from Kubernetes secrets):
  - Username: `postgres`
  - Password: `dfnks.irfheaei;vnc.nvdfighnsnfncxvisruhn`
- **Configuration**:
  ```bash
  POSTGRES_HOST=pg-haproxy-primary.pg.svc.cluster.local
  POSTGRES_PORT=5432
  POSTGRES_USER=postgres
  POSTGRES_PASSWORD=dfnks.irfheaei;vnc.nvdfighnsnfncxvisruhn
  POSTGRES_DB=postgres
  ```

### 4. Neo4j Graph Database
- **Service**: `neo4j-clusterip.neo4j.svc.cluster.local:7687` (Bolt)
- **HTTP Service**: `neo4j-clusterip.neo4j.svc.cluster.local:7474`
- **External Access**: LoadBalancer with multiple IPs
- **Purpose**: Graph relationships and knowledge graph storage
- **Credentials** (from Kubernetes secrets):
  - Username: `neo4j`
  - Password: `dCqNHU1sgz99lF7h`
- **Configuration**:
  ```bash
  NEO4J_URI=bolt://neo4j-clusterip.neo4j.svc.cluster.local:7687
  NEO4J_AUTH=neo4j/dCqNHU1sgz99lF7h
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=dCqNHU1sgz99lF7h
  ```

### 5. Redis Cache
- **Service**: `redis-cluster-headless.redis.svc.cluster.local:6379` (Headless service for Redis Enterprise cluster)
- **Purpose**: Caching and session management
- **Credentials** (from Kubernetes secrets):
  - Password: `fedfina_staging_redis_password_secure`
- **Configuration**:
  ```bash
  REDIS_HOST=redis-cluster-headless.redis.svc.cluster.local
  REDIS_PORT=6379
  REDIS_PASSWORD=fedfina_staging_redis_password_secure
  REDIS_DB=0
  ```

### 6. AI Infrastructure Services
- **Base Host**: `192.168.0.20` (External AI services)
- **Purpose**: External AI/ML services to avoid heavy local dependencies

#### 6.1 vLLM Service (LLM Inference)
- **Service**: `192.168.0.20:8000`
- **Purpose**: Large Language Model inference
- **Configuration**:
  ```bash
  VLLM_BASE_URL=http://192.168.0.20:8000
  VLLM_MODEL_NAME=default
  ```

#### 6.2 Routing API
- **Service**: `192.168.0.20:8001`
- **Purpose**: API routing and load balancing
- **Configuration**:
  ```bash
  AI_ROUTING_API_URL=http://192.168.0.20:8001
  ```

#### 6.3 Speech-to-Text Service
- **Service**: `192.168.0.20:8002`
- **Purpose**: Audio transcription
- **Configuration**:
  ```bash
  STT_SERVICE_URL=http://192.168.0.20:8002
  ```

#### 6.4 Text-to-Speech Service
- **Service**: `192.168.0.20:8003`
- **Purpose**: Audio generation
- **Configuration**:
  ```bash
  TTS_SERVICE_URL=http://192.168.0.20:8003
  ```

## Complete Environment File

Create a `.env` file with the following configuration:

```bash
# =============================================================================
# AGENTIC MULTIMODAL RAG SYSTEM - KUBERNETES ENVIRONMENT CONFIGURATION
# =============================================================================

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
MINIO_ACCESS_KEY=x77at4PB02HDuMNXNwr2
MINIO_SECRET_KEY=LY0MkJ1Vawto8K8X4ICqlJ3Drm5I5AzezkWPLztE
MINIO_SECURE=false
MINIO_BUCKET=rag-docs
MINIO_REGION=us-east-1

# =============================================================================
# POSTGRESQL DATABASE CONFIGURATION
# =============================================================================
POSTGRES_HOST=pg-haproxy-primary.pg.svc.cluster.local
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=dfnks.irfheaei;vnc.nvdfighnsnfncxvisruhn
POSTGRES_DB=postgres
POSTGRES_SSL=false
POSTGRES_POOL_MIN=2
POSTGRES_POOL_MAX=10
POSTGRES_CONNECTION_TIMEOUT=60000

# Database URL (alternative format)
DATABASE_URL=postgresql://postgres:dfnks.irfheaei;vnc.nvdfighnsnfncxvisruhn@pg-haproxy-primary.pg.svc.cluster.local:5432/postgres

# =============================================================================
# REDIS CONFIGURATION
# =============================================================================
REDIS_HOST=redis-cluster-headless.redis.svc.cluster.local
REDIS_PORT=6379
REDIS_PASSWORD=fedfina_staging_redis_password_secure
REDIS_DB=0
REDIS_URL=redis://:fedfina_staging_redis_password_secure@redis-cluster-headless.redis.svc.cluster.local:6379/0

# Redis Connection Pool
REDIS_POOL_SIZE=10
REDIS_CONNECT_TIMEOUT=10000
REDIS_COMMAND_TIMEOUT=5000

# =============================================================================
# NEO4J GRAPH DATABASE CONFIGURATION
# =============================================================================
NEO4J_URI=bolt://neo4j.neo4j.svc.cluster.local:7687
NEO4J_AUTH=neo4j/dCqNHU1sgz99lF7h
NEO4J_USER=neo4j
NEO4J_PASSWORD=dCqNHU1sgz99lF7h

# Neo4j HTTP endpoint (for browser access)
  NEO4J_HTTP_URI=http://neo4j-clusterip.neo4j.svc.cluster.local:7474

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
```

## External Access Options

If you need to access services from outside the cluster, use these external endpoints:

### Milvus
- **External**: `192.168.0.204:19530`

### MinIO
- **External**: `192.168.0.204:30900` (NodePort)
- **Console**: `192.168.0.204:32299` (NodePort)

### Neo4j
- **Bolt**: `192.168.0.20:7687` (LoadBalancer)
- **HTTP**: `192.168.0.20:7474` (LoadBalancer)

### PostgreSQL
- **HAProxy Primary**: `pg-haproxy-primary.pg.svc.cluster.local:5432` (LoadBalancer)
- **External Access**: `192.168.0.212:5432` (via LoadBalancer IP)
- **Internal Access**: Use HAProxy service for automatic failover and load balancing
- **Port Forwarding** (alternative):
  ```bash
  kubectl port-forward -n pg svc/pg-haproxy-primary 5432:5432
  ```

### Redis
- **Internal Only**: Use port-forwarding for external access
  ```bash
  kubectl port-forward -n redis svc/redis-cluster-headless 6379:6379
  ```

## Security Notes

1. **Credentials**: All credentials are extracted from Kubernetes secrets and are production-ready
2. **Network**: Services use internal cluster networking for security
3. **SSL/TLS**: Configure SSL for production deployments
4. **Access Control**: Implement proper RBAC for Kubernetes resources

## Testing Connectivity

You can test the connectivity using the health check endpoints:

```bash
# Test all services
curl http://localhost:8000/health

# Test detailed service status
curl http://localhost:8000/health/details
```

## Next Steps

1. Copy the environment configuration to a `.env` file
2. Update any placeholder values (JWT secrets, API keys, etc.)
3. Deploy the application to the Kubernetes cluster
4. Verify connectivity using the health check endpoints
5. Test the full RAG pipeline with sample documents
