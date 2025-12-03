# Kubernetes Deployment - Agentic Multimodal RAG System

This directory contains all Kubernetes manifests and documentation for deploying the Agentic Multimodal RAG System to a Kubernetes cluster.

## 📁 Directory Structure

```
k8s/
├── README.md                    # This file - overview and quick reference
├── DEPLOYMENT.md                # Comprehensive deployment guide
├── namespace.yaml               # Namespace definition
├── rbac.yaml                    # ServiceAccount, Role, RoleBinding
├── pvc.yaml                     # PersistentVolumeClaims (models & logs)
├── secret.yaml                  # Application secrets (credentials)
├── configmap.yaml              # Application configuration
├── service.yaml                 # Kubernetes services
├── deployment.yaml              # Main application deployment
├── ingress.yaml                 # Nginx ingress configuration
├── kong-ingress.yaml            # Kong ingress configuration
└── deploy.sh                    # Deployment automation script
```

## 🚀 Quick Start

### Prerequisites
- `kubectl` configured and connected to your cluster
- Cluster-admin or namespace admin permissions for the `rag` namespace
- Docker image available: `registry.bionicaisolutions.com/rag/agentic-multimodal-rag:latest`

### Deploy Everything

```bash
# Option 1: Using the deployment script (Recommended)
cd /workspace
./k8s/deploy.sh

# Option 2: Manual deployment
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/deployment.yaml
```

### Verify Deployment

```bash
# Check pod status
kubectl get pods -n rag

# Check services
kubectl get svc -n rag

# View logs
kubectl logs -n rag -l app.kubernetes.io/name=agentic-multimodal-rag -f
```

## 📋 Configuration Summary

### Service Dependencies

The application connects to these cluster services:

| Service | DNS Endpoint | Purpose |
|---------|-------------|---------|
| **PostgreSQL** | `pg-ceph-rw.pg.svc.cluster.local:5432` | Primary database |
| **MinIO** | `minio-tenant-hl.minio.svc.cluster.local:9000` | Object storage |
| **Milvus** | `milvus.milvus.svc.cluster.local:19530` | Vector database |
| **Redis** | `redis-cluster.redis.svc.cluster.local:6379` | Cache |
| **Neo4j** | `neo4j-clusterip.neo4j.svc.cluster.local:7687` | Graph database |

### Storage Configuration

| PVC | Size | Storage Class | Access Mode | Purpose |
|-----|------|---------------|-------------|---------|
| `rag-models-pvc` | 50Gi | `cephfs` | ReadWriteMany | AI model storage |
| `rag-logs-pvc` | 10Gi | `cephfs` | ReadWriteMany | Persistent logs |

### Resource Requirements

- **Replicas**: 3
- **CPU per pod**: 1000m request, 2000m limit
- **Memory per pod**: 2Gi request, 4Gi limit
- **Total**: 3000m CPU, 6Gi memory (requests)

## 📚 Documentation

- **[DEPLOYMENT.md](./DEPLOYMENT.md)** - Comprehensive deployment guide with troubleshooting
- **[env-reference.md](./env-reference.md)** - Complete environment variables reference

## 🔧 Common Operations

### Update Configuration

```bash
# Update ConfigMap
kubectl apply -f k8s/configmap.yaml
kubectl rollout restart deployment/rag-app -n rag

# Update Secrets
kubectl apply -f k8s/secret.yaml
kubectl rollout restart deployment/rag-app -n rag
```

### Scale Deployment

```bash
# Scale to 5 replicas
kubectl scale deployment rag-app -n rag --replicas=5
```

### Access Application

```bash
# Port forward for local access
kubectl port-forward -n rag svc/rag-app-service 8000:80

# Access endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

### Cleanup

```bash
# Delete all resources
kubectl delete -f k8s/
```

## 🔐 Security Notes

- All secrets are base64 encoded in `secret.yaml`
- Container runs as non-root user (UID 1000)
- RBAC configured with least privilege
- Image pull secrets configured for private registry

## 📝 Notes

- All service endpoints use Kubernetes internal DNS
- Storage uses Ceph (CephFS for shared volumes, Ceph RBD for fast storage)
- Health checks configured on `/health` endpoint
- Logs are persisted to CephFS volume

For detailed deployment instructions, see [DEPLOYMENT.md](./DEPLOYMENT.md).

