# Kubernetes Environment Variables Reference

This document provides a complete reference of all environment variables configured for the Agentic Multimodal RAG System deployment in the Kubernetes cluster (namespace: `rag`).

## Service Endpoints

All services use Kubernetes internal DNS names for service discovery within the cluster.

### Milvus Vector Database
- **Service**: `milvus.milvus.svc.cluster.local:19530`
- **ConfigMap Variables**:
  - `MILVUS_HOST=milvus.milvus.svc.cluster.local`
  - `MILVUS_PORT=19530`

### MinIO Object Storage
- **Service**: `minio.minio.svc.cluster.local:80`
- **ConfigMap Variables**:
  - `MINIO_HOST=minio.minio.svc.cluster.local`
  - `MINIO_PORT=80`
  - `MINIO_SECURE=false`
  - `MINIO_BUCKET=rag-docs`
  - `MINIO_REGION=us-east-1`
- **Secret Variables** (from `rag-app-secrets`):
  - `MINIO_ACCESS_KEY` (base64 encoded)
  - `MINIO_SECRET_KEY` (base64 encoded)

### PostgreSQL Database
- **Service**: `pg-rw.pg.svc.cluster.local:5432`
- **ConfigMap Variables**:
  - `POSTGRES_HOST=pg-rw.pg.svc.cluster.local`
  - `POSTGRES_PORT=5432`
  - `POSTGRES_USER=postgres`
  - `POSTGRES_DB=postgres`
  - `POSTGRES_SSL=false`
  - `POSTGRES_POOL_MIN=2`
  - `POSTGRES_POOL_MAX=10`
  - `POSTGRES_CONNECTION_TIMEOUT=60000`
- **Secret Variables** (from `rag-app-secrets`):
  - `POSTGRES_PASSWORD` (base64 encoded)

### Redis Cache
- **Service**: `redis-simple.redis-new.svc.cluster.local:6379`
- **ConfigMap Variables**:
  - `REDIS_HOST=redis-simple.redis-new.svc.cluster.local`
  - `REDIS_PORT=6379`
  - `REDIS_DB=0`
  - `REDIS_POOL_SIZE=10`
  - `REDIS_CONNECT_TIMEOUT=10000`
  - `REDIS_COMMAND_TIMEOUT=5000`
- **Secret Variables** (from `rag-app-secrets`):
  - `REDIS_PASSWORD` (base64 encoded)

### Neo4j Graph Database
- **Bolt Service**: `neo4j.neo4j.svc.cluster.local:7687`
- **HTTP Service**: `neo4j.neo4j.svc.cluster.local:7474`
- **ConfigMap Variables**:
  - `NEO4J_URI=bolt://neo4j.neo4j.svc.cluster.local:7687`
  - `NEO4J_USER=neo4j`
  - `NEO4J_HTTP_URI=http://neo4j.neo4j.svc.cluster.local:7474`
- **Secret Variables** (from `rag-app-secrets`):
  - `NEO4J_PASSWORD` (base64 encoded)
  - `NEO4J_AUTH` (base64 encoded)

## AI Infrastructure Services

External AI services running on dedicated infrastructure:

- **vLLM Service**: `http://192.168.0.20:8000`
- **Routing API**: `http://192.168.0.20:8001`
- **STT Service**: `http://192.168.0.20:8002`
- **TTS Service**: `http://192.168.0.20:8003`

## LLM Backend Configuration

- `LLM_BACKEND=local` (uses Ollama)
- `OLLAMA_HOST=192.168.0.199`
- `OLLAMA_PORT=11434`
- `OLLAMA_MODEL=deepseek-r1:8b`

## FastAPI MCP Configuration

- `MCP_BASE_URL=http://rag-app-internal.rag.svc.cluster.local:8000/mcp`

This enables internal FastAPI MCP tool calls within the cluster.

## Model Storage

Models are stored in a PersistentVolume mounted at `/opt/ai-models`:
- `HF_HOME=/opt/ai-models`
- `TRANSFORMERS_CACHE=/opt/ai-models`
- `MODEL_DIR=/opt/ai-models`

## Deployment

The environment variables are loaded from:
1. **ConfigMap**: `rag-app-config` (non-sensitive configuration)
2. **Secret**: `rag-app-secrets` (sensitive credentials)

Both are loaded via `envFrom` in the deployment, with Secrets taking precedence for duplicate keys.

## Updating Configuration

To update the configuration:

1. **Update ConfigMap** (for non-sensitive values):
   ```bash
   kubectl edit configmap rag-app-config -n rag
   ```

2. **Update Secret** (for sensitive values):
   ```bash
   kubectl edit secret rag-app-secrets -n rag
   ```

3. **Restart pods** to pick up changes:
   ```bash
   kubectl rollout restart deployment rag-app -n rag
   ```

## Verification

Check that environment variables are loaded correctly:

```bash
# Check ConfigMap
kubectl get configmap rag-app-config -n rag -o yaml

# Check Secret (values will be base64 encoded)
kubectl get secret rag-app-secrets -n rag -o yaml

# Check environment variables in a running pod
kubectl exec -n rag deployment/rag-app -- env | grep -E "(MILVUS|MINIO|POSTGRES|REDIS|NEO4J)"
```

