# Deployment Guide - Agentic Multimodal RAG System

This guide provides comprehensive instructions for deploying the Agentic Multimodal RAG System using GitHub Actions, Docker, and Kubernetes.

## 🚀 Quick Start

### Prerequisites

1. **GitHub Repository** with the following secrets configured:
   - `REGISTRY_USERNAME`: Your registry username
   - `REGISTRY_TOKEN`: Your registry access token

2. **Kubernetes Cluster** with the following services available:
   - Milvus (Vector Database)
   - MinIO (Object Storage)
   - PostgreSQL (Relational Database)
   - Neo4j (Graph Database)
   - Redis (Cache)

3. **Registry Access** to `registry.bionicaisolutions.com` for image storage

4. **Cluster Access** via one of:
   - Rancher UI (recommended for internet-accessible clusters)
   - Direct kubectl access
   - Rancher CLI

### Automatic Build via GitHub Actions

The system includes a CI/CD pipeline that automatically:

1. **Runs Tests**: Unit tests, linting, security scanning
2. **Builds Docker Image**: Multi-architecture builds (AMD64, ARM64)
3. **Pushes to Registry**: Tagged with version and latest to `registry.bionicaisolutions.com/rag/agentic-multimodal-rag`
4. **Provides Deployment Instructions**: Next steps for manual deployment

#### Triggering Build

```bash
# Build and push latest version
git push origin main

# Build and push specific version
git tag v1.0.0
git push origin v1.0.0
```

**Note**: The GitHub Actions workflow will build and push Docker images to `registry.bionicaisolutions.com`, but will not automatically deploy to your K3s cluster. You'll need to deploy manually using one of the methods below.

## 🐳 Docker Deployment

### Local Development

```bash
# Build and run with docker-compose
docker-compose up -d

# Build and run production version
docker-compose -f docker-compose.prod.yml up -d
```

### Manual Docker Build

```bash
# Build and push using the build script (recommended)
./scripts/build-and-push.sh latest

# Or manually build and push
docker build -t registry.bionicaisolutions.com/rag/agentic-multimodal-rag:latest .
docker login registry.bionicaisolutions.com
docker push registry.bionicaisolutions.com/rag/agentic-multimodal-rag:latest

# Run the container locally
docker run -p 8000:8000 --env-file .env registry.bionicaisolutions.com/rag/agentic-multimodal-rag:latest
```

## ☸️ Kubernetes Deployment

### Method 1: Rancher UI Deployment (Recommended)

Since your K3s cluster is not directly accessible from the internet, use Rancher UI for deployment:

1. **Access Rancher UI** in your web browser
2. **Navigate to your K3s cluster**
3. **Create namespace**: `rag-system`
4. **Apply manifests** using the Rancher UI:
   - ConfigMap: `k8s/configmap.yaml`
   - Secrets: `k8s/secret.yaml`
   - PVC: `k8s/pvc.yaml`
   - Deployment: `k8s/deployment.yaml` (update image tag)
   - Services: `k8s/service.yaml`
   - Ingress: `k8s/ingress.yaml`

For detailed Rancher deployment instructions, see: [Rancher Deployment Guide](Rancher-Deployment-Guide.md)

### Method 2: Using the Rancher Deployment Script

```bash
# Deploy via Rancher CLI (if available)
./scripts/deploy-rancher.sh deploy latest

# Deploy with Rancher URL
RANCHER_URL=https://your-rancher-url ./scripts/deploy-rancher.sh deploy latest

# Check deployment status
./scripts/deploy-rancher.sh status

# View logs
./scripts/deploy-rancher.sh logs

# Run health check
./scripts/deploy-rancher.sh health
```

### Method 3: Direct kubectl Deployment

If you have direct kubectl access to your cluster:

```bash
# Deploy latest version
./scripts/deploy.sh deploy latest

# Deploy specific version
./scripts/deploy.sh deploy v1.0.0

# Check deployment status
./scripts/deploy.sh status

# View logs
./scripts/deploy.sh logs

# Run health check
./scripts/deploy.sh health

# Cleanup deployment
./scripts/deploy.sh cleanup
```

### Manual Kubernetes Deployment

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply RBAC
kubectl apply -f k8s/rbac.yaml

# Create persistent volumes
kubectl apply -f k8s/pvc.yaml

# Apply configuration
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml

# Expose services
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

## 📋 Configuration

### Environment Variables

The system uses the following configuration sources (in order of precedence):

1. **Kubernetes Secrets** (for sensitive data)
2. **Kubernetes ConfigMaps** (for configuration)
3. **Environment Variables** (for overrides)

### Key Configuration Files

- **`.env`**: Local development environment
- **`k8s/configmap.yaml`**: Kubernetes configuration
- **`k8s/secret.yaml`**: Kubernetes secrets
- **`docker-compose.prod.yml`**: Production Docker setup

### Service Dependencies

The application requires the following services:

| Service | Purpose | Port | Namespace |
|---------|---------|------|-----------|
| Milvus | Vector Database | 19530 | milvus |
| MinIO | Object Storage | 80 | minio |
| PostgreSQL | Relational DB | 5432 | pg |
| Neo4j | Graph Database | 7687 | neo4j |
| Redis | Cache | 6379 | redis-new |

## 🔧 Customization

### Updating Docker Image

1. **Modify Dockerfile**: Update the Dockerfile for application changes
2. **Update Dependencies**: Modify `pyproject.toml` for Python dependencies
3. **Rebuild**: The CI/CD pipeline will automatically rebuild on push

### Scaling the Application

```bash
# Scale horizontally
kubectl scale -n rag-system deployment/rag-app --replicas=5

# Update resource limits
kubectl patch -n rag-system deployment/rag-app -p '{"spec":{"template":{"spec":{"containers":[{"name":"rag-app","resources":{"limits":{"memory":"8Gi","cpu":"4000m"}}}]}}}}'
```

### Custom Resource Configuration

```bash
# Update ConfigMap
kubectl edit -n rag-system configmap rag-app-config

# Update Secrets
kubectl edit -n rag-system secret rag-app-secrets

# Restart deployment to apply changes
kubectl rollout restart -n rag-system deployment/rag-app
```

## 🔍 Monitoring and Troubleshooting

### Health Checks

The application provides several health check endpoints:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/details
```

### Logging

```bash
# View application logs
kubectl logs -n rag-system -l app.kubernetes.io/name=agentic-multimodal-rag -f

# View specific pod logs
kubectl logs -n rag-system deployment/rag-app -f
```

### Common Issues

#### 1. Pod Startup Issues

```bash
# Check pod status
kubectl get pods -n rag-system

# Describe pod for details
kubectl describe pod -n rag-system <pod-name>

# Check events
kubectl get events -n rag-system --sort-by='.lastTimestamp'
```

#### 2. Service Connectivity

```bash
# Test service connectivity
kubectl exec -n rag-system deployment/rag-app -- curl -f http://localhost:8000/health

# Check service endpoints
kubectl get endpoints -n rag-system
```

#### 3. Resource Issues

```bash
# Check resource usage
kubectl top pods -n rag-system
kubectl top nodes

# Check resource limits
kubectl describe pod -n rag-system <pod-name>
```

## 🔐 Security Considerations

### Secrets Management

- All sensitive data is stored in Kubernetes secrets
- Secrets are base64 encoded (not encrypted)
- Use external secret management for production

### Network Security

- Services use internal cluster networking
- LoadBalancer provides external access
- Ingress controller handles SSL termination

### RBAC

- Service account with minimal required permissions
- Role-based access control for Kubernetes resources
- Non-root container execution

## 📊 Performance Optimization

### Resource Tuning

```yaml
# Example resource configuration
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Horizontal Pod Autoscaling

```bash
# Create HPA
kubectl autoscale -n rag-system deployment/rag-app --cpu-percent=70 --min=3 --max=10
```

### Caching Strategy

- Redis for session and application caching
- Model caching in persistent volumes
- CDN for static assets (if applicable)

## 🚨 Backup and Recovery

### Data Backup

```bash
# Backup persistent volumes
kubectl exec -n rag-system deployment/rag-app -- tar -czf /tmp/backup.tar.gz /opt/ai-models

# Backup configuration
kubectl get configmap rag-app-config -n rag-system -o yaml > config-backup.yaml
kubectl get secret rag-app-secrets -n rag-system -o yaml > secrets-backup.yaml
```

### Disaster Recovery

1. **Restore from backup**: Apply backed up configurations
2. **Redeploy**: Use the deployment script to restore services
3. **Verify**: Run health checks to ensure system integrity

## 📈 Monitoring and Alerting

### Metrics Collection

The application exposes metrics on `/metrics` endpoint for Prometheus scraping.

### Log Aggregation

Logs are structured in JSON format and can be collected by:
- Fluentd/Fluent Bit
- ELK Stack
- Splunk
- Cloud logging services

### Alerting Rules

Set up alerts for:
- High CPU/Memory usage
- Pod restart frequency
- Health check failures
- Service availability

## 🔄 Updates and Maintenance

### Rolling Updates

```bash
# Update image tag
kubectl set image -n rag-system deployment/rag-app rag-app=your-username/agentic-multimodal-rag:v1.1.0

# Monitor rollout
kubectl rollout status -n rag-system deployment/rag-app
```

### Maintenance Windows

```bash
# Scale down for maintenance
kubectl scale -n rag-system deployment/rag-app --replicas=0

# Perform maintenance tasks
# ...

# Scale back up
kubectl scale -n rag-system deployment/rag-app --replicas=3
```

## 📞 Support

For issues and support:

1. Check the logs and health endpoints
2. Review the troubleshooting section
3. Check GitHub Issues for known problems
4. Contact the development team

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Milvus Documentation](https://milvus.io/docs/)
- [MinIO Documentation](https://docs.min.io/)
