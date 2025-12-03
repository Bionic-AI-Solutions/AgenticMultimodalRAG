# Kubernetes Deployment Guide

This guide explains how to deploy the Agentic Multimodal RAG System to your Kubernetes cluster.

## Prerequisites

1. **Kubernetes Access**: You need cluster-admin or namespace admin permissions for the `rag` namespace
2. **kubectl**: Kubernetes CLI tool configured and connected to your cluster
3. **Docker Image**: The application image `registry.bionicaisolutions.com/rag/agentic-multimodal-rag:latest` should be available

## Quick Deployment

### Option 1: Using the Deployment Script (Recommended)

```bash
cd /workspace/AgenticMultimodalRAG
./k8s/deploy.sh
```

### Option 2: Manual Step-by-Step Deployment

Deploy resources in the following order:

```bash
# 1. Create namespace
kubectl apply -f k8s/namespace.yaml

# 2. Create RBAC (ServiceAccount, Role, RoleBinding)
kubectl apply -f k8s/rbac.yaml

# 3. Create PersistentVolumeClaims (for models and logs)
kubectl apply -f k8s/pvc.yaml

# 4. Create Secrets (credentials)
kubectl apply -f k8s/secret.yaml

# 5. Create ConfigMap (configuration)
kubectl apply -f k8s/configmap.yaml

# 6. Create Services
kubectl apply -f k8s/service.yaml

# 7. Deploy Application
kubectl apply -f k8s/deployment.yaml
```

## Verify Deployment

### Check Pod Status

```bash
kubectl get pods -n rag
```

Expected output: 3 pods in `Running` state (based on `replicas: 3`)

### Check Services

```bash
kubectl get svc -n rag
```

You should see:
- `rag-app-service` (LoadBalancer) - External access
- `rag-app-internal` (ClusterIP) - Internal cluster access

### Check Deployment

```bash
kubectl get deployment -n rag
```

### View Pod Logs

```bash
# View logs from all pods
kubectl logs -n rag -l app.kubernetes.io/name=agentic-multimodal-rag -f

# View logs from a specific pod
kubectl logs -n rag <pod-name> -f
```

### Check Pod Events

```bash
kubectl describe pod -n rag -l app.kubernetes.io/name=agentic-multimodal-rag
```

## Access the Application

### Internal Access (from within cluster)

- **Internal Service**: `http://rag-app-internal.rag.svc.cluster.local:8000`
- **MCP Endpoint**: `http://rag-app-internal.rag.svc.cluster.local:8000/mcp`
- **Health Check**: `http://rag-app-internal.rag.svc.cluster.local:8000/health`
- **API Docs**: `http://rag-app-internal.rag.svc.cluster.local:8000/docs`

### External Access

1. **Via LoadBalancer Service**:
   ```bash
   kubectl get svc rag-app-service -n rag
   ```
   Use the `EXTERNAL-IP` to access the application.

2. **Via Port Forwarding** (for testing):
   ```bash
   kubectl port-forward -n rag svc/rag-app-service 8000:80
   ```
   Then access: `http://localhost:8000`

## Troubleshooting

### Pods Not Starting

1. **Check pod status**:
   ```bash
   kubectl get pods -n rag
   kubectl describe pod <pod-name> -n rag
   ```

2. **Check pod logs**:
   ```bash
   kubectl logs <pod-name> -n rag
   ```

3. **Common issues**:
   - **ImagePullBackOff**: Check if the Docker image exists and is accessible
   - **CrashLoopBackOff**: Check application logs for errors
   - **Pending**: Check PVC status and node resources

### PVC Issues

```bash
# Check PVC status
kubectl get pvc -n rag

# Check PV status
kubectl get pv | grep rag
```

### Configuration Issues

```bash
# Verify ConfigMap
kubectl get configmap rag-app-config -n rag -o yaml

# Verify Secrets (values will be base64 encoded)
kubectl get secret rag-app-secrets -n rag -o yaml
```

### Service Connectivity

```bash
# Test service from within cluster
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -n rag -- \
  curl http://rag-app-internal.rag.svc.cluster.local:8000/health
```

## Scaling

### Scale Up/Down

```bash
# Scale to 5 replicas
kubectl scale deployment rag-app -n rag --replicas=5

# Scale to 1 replica
kubectl scale deployment rag-app -n rag --replicas=1
```

### Update Deployment

```bash
# Update image
kubectl set image deployment/rag-app rag-app=registry.bionicaisolutions.com/rag/agentic-multimodal-rag:new-tag -n rag

# Restart deployment
kubectl rollout restart deployment/rag-app -n rag
```

## Rollback

```bash
# Check rollout history
kubectl rollout history deployment/rag-app -n rag

# Rollback to previous version
kubectl rollout undo deployment/rag-app -n rag

# Rollback to specific revision
kubectl rollout undo deployment/rag-app -n rag --to-revision=2
```

## Cleanup

To remove all resources:

```bash
# Delete deployment and services
kubectl delete -f k8s/deployment.yaml
kubectl delete -f k8s/service.yaml

# Delete ConfigMap and Secrets
kubectl delete -f k8s/configmap.yaml
kubectl delete -f k8s/secret.yaml

# Delete PVCs (WARNING: This will delete data)
kubectl delete -f k8s/pvc.yaml

# Delete RBAC
kubectl delete -f k8s/rbac.yaml

# Delete namespace (WARNING: This deletes everything)
kubectl delete -f k8s/namespace.yaml
```

Or delete everything at once:

```bash
kubectl delete -f k8s/
```

## Configuration Updates

After updating ConfigMap or Secrets:

```bash
# Apply changes
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Restart pods to pick up changes
kubectl rollout restart deployment/rag-app -n rag
```

## Service Dependencies

The application connects to these services (configured in ConfigMap):

- **PostgreSQL**: `pg-ceph-rw.pg.svc.cluster.local:5432` (read-write access)
- **MinIO**: `minio-tenant-hl.minio.svc.cluster.local:9000` (headless service)
- **Milvus**: `milvus.milvus.svc.cluster.local:19530`
- **Redis**: `redis-cluster.redis.svc.cluster.local:6379` (load-balanced)
- **Neo4j**: `neo4j-clusterip.neo4j.svc.cluster.local:7687` (Bolt) and `:7474` (HTTP)

### Storage Configuration

- **Models PVC**: 50Gi using `cephfs` storage class (ReadWriteMany)
- **Logs PVC**: 10Gi using `cephfs` storage class (ReadWriteMany)
- Both volumes use Ceph persistent storage for high performance and reliability

Ensure these services are running and accessible from the `rag` namespace.

