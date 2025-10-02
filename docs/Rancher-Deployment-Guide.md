# Rancher Deployment Guide - Agentic Multimodal RAG System

This guide provides instructions for deploying the Agentic Multimodal RAG System using Rancher UI, which is accessible through the internet and can manage your K3s cluster.

## 🌐 Rancher Deployment Overview

Since your K3s cluster is not directly accessible from the internet, we can use Rancher (which is accessible) to deploy and manage the application. Rancher provides a web UI that can manage your K3s cluster remotely.

## 📋 Prerequisites

1. **Rancher Access**: Rancher UI accessible via internet
2. **K3s Cluster**: Connected to Rancher
3. **Docker Hub**: Images pushed via GitHub Actions
4. **Required Services**: Milvus, MinIO, PostgreSQL, Neo4j, Redis in your cluster

## 🚀 Deployment Methods

### Method 1: Rancher UI Deployment (Recommended)

#### Step 1: Access Rancher UI
1. Open your Rancher UI in a web browser
2. Navigate to your K3s cluster
3. Go to **Apps & Marketplace** or **Workloads**

#### Step 2: Create Namespace
1. Go to **Cluster Explorer** → **Namespaces**
2. Click **Create**
3. Name: `rag-system`
4. Add labels:
   - `app.kubernetes.io/name: agentic-multimodal-rag`
   - `app.kubernetes.io/version: 1.0.0`

#### Step 3: Create ConfigMap
1. Go to **Cluster Explorer** → **ConfigMaps**
2. Click **Create**
3. Name: `rag-app-config`
4. Namespace: `rag-system`
5. Copy the content from `k8s/configmap.yaml` into the YAML editor

#### Step 4: Create Secrets
1. Go to **Cluster Explorer** → **Secrets**
2. Click **Create**
3. Name: `rag-app-secrets`
4. Namespace: `rag-system`
5. Type: `Opaque`
6. Copy the content from `k8s/secret.yaml` into the YAML editor

#### Step 5: Create Persistent Volume Claims
1. Go to **Cluster Explorer** → **Storage** → **Persistent Volume Claims**
2. Click **Create**
3. Use the content from `k8s/pvc.yaml`

#### Step 6: Deploy Application
1. Go to **Cluster Explorer** → **Workloads** → **Deployments**
2. Click **Create**
3. Use the content from `k8s/deployment.yaml`
4. **Important**: Update the image tag to your Docker Hub image:
   ```yaml
   image: your-docker-username/agentic-multimodal-rag:latest
   ```

#### Step 7: Create Services
1. Go to **Cluster Explorer** → **Service Discovery** → **Services**
2. Click **Create**
3. Use the content from `k8s/service.yaml`

#### Step 8: Create Ingress (Optional)
1. Go to **Cluster Explorer** → **Service Discovery** → **Ingress**
2. Click **Create**
3. Use the content from `k8s/ingress.yaml`

### Method 2: Rancher CLI Deployment

If you have Rancher CLI access:

```bash
# Login to Rancher
rancher login https://your-rancher-url --token your-rancher-token

# Set context to your cluster
rancher context switch your-cluster-id

# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

### Method 3: Rancher Apps & Marketplace

#### Create Custom App
1. Go to **Apps & Marketplace** → **Create**
2. Choose **Custom App**
3. Name: `agentic-multimodal-rag`
4. Namespace: `rag-system`
5. Upload or paste the Kubernetes manifests

## 🔧 Configuration for Rancher

### Required Rancher Information

To help you set up the deployment, I need to know:

1. **Rancher URL**: What's your Rancher web interface URL?
2. **Cluster Name**: What's the name of your K3s cluster in Rancher?
3. **Docker Hub Username**: Your Docker Hub username for the image
4. **Access Method**: Do you prefer UI, CLI, or API access?

### Rancher-Specific Configurations

#### Service Account and RBAC
```yaml
# Apply RBAC first
kubectl apply -f k8s/rbac.yaml
```

#### Resource Limits
Adjust based on your cluster capacity:
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

#### Storage Classes
Update the PVC to use your cluster's storage class:
```yaml
storageClassName: nfs-client  # or your cluster's storage class
```

## 📊 Monitoring in Rancher

### View Application Status
1. Go to **Cluster Explorer** → **Workloads**
2. Filter by namespace: `rag-system`
3. Check deployment status and pod health

### View Logs
1. Click on the deployment
2. Go to **Pods** tab
3. Click on a pod → **View Logs**

### Monitor Resources
1. Go to **Cluster Explorer** → **Monitoring**
2. View CPU, Memory, and Network usage

## 🔄 Updates via Rancher

### Update Application
1. Go to **Workloads** → **Deployments**
2. Find `rag-app` deployment
3. Click **Edit**
4. Update the image tag to the new version
5. Click **Save**

### Rollback if Needed
1. Go to **Workloads** → **Deployments**
2. Click on `rag-app`
3. Go to **History** tab
4. Click **Rollback** on the previous version

## 🚨 Troubleshooting

### Common Issues

#### 1. Image Pull Errors
```bash
# Check if image exists in Docker Hub
docker pull your-username/agentic-multimodal-rag:latest

# Check image pull secrets
kubectl get secrets -n rag-system
```

#### 2. Pod Startup Issues
- Check pod logs in Rancher UI
- Verify ConfigMap and Secret are applied
- Check resource limits and requests

#### 3. Service Connectivity
- Verify service endpoints
- Check network policies
- Test internal connectivity

### Health Checks
```bash
# Port forward to test locally
kubectl port-forward -n rag-system svc/rag-app-internal 8080:8000

# Test health endpoint
curl http://localhost:8080/health
```

## 🔐 Security Considerations

### Rancher Security
1. **RBAC**: Use Rancher's built-in RBAC
2. **Network Policies**: Configure network policies
3. **Pod Security**: Use security contexts
4. **Secrets Management**: Store sensitive data in Kubernetes secrets

### Access Control
1. **User Management**: Use Rancher's user management
2. **Project Access**: Limit access to specific namespaces
3. **API Access**: Use Rancher API tokens

## 📈 Scaling in Rancher

### Horizontal Scaling
1. Go to **Workloads** → **Deployments**
2. Click on `rag-app`
3. Click **Scale**
4. Set desired replica count

### Vertical Scaling
1. Edit the deployment
2. Update resource limits
3. Save changes

## 🔄 CI/CD Integration with Rancher

### Webhook Integration
You can set up webhooks in Rancher to automatically deploy when new images are pushed:

1. Go to **Apps & Marketplace** → **Webhooks**
2. Create webhook for image updates
3. Configure to trigger deployment updates

### GitOps with Rancher
1. Use Rancher's GitOps capabilities
2. Connect to your Git repository
3. Automatically sync changes

## 📞 Support and Next Steps

### What I Need from You

To provide more specific guidance, please share:

1. **Rancher URL**: Your Rancher web interface URL
2. **Cluster Details**: K3s cluster name and configuration
3. **Docker Hub**: Your Docker Hub username
4. **Preferred Method**: UI, CLI, or API deployment preference

### Immediate Actions

1. **Push to GitHub**: Trigger the CI/CD pipeline to build and push Docker images
2. **Access Rancher**: Log into your Rancher UI
3. **Prepare Manifests**: Have the Kubernetes manifests ready
4. **Test Deployment**: Start with a small deployment to test connectivity

### Future Enhancements

1. **Automated Deployment**: Set up webhooks for automatic deployment
2. **Monitoring**: Configure comprehensive monitoring
3. **Backup**: Set up backup strategies
4. **Security**: Implement security best practices

## 📚 Additional Resources

- [Rancher Documentation](https://rancher.com/docs/)
- [K3s Documentation](https://k3s.io/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)

---

**Ready to deploy?** Let me know your Rancher details and I can provide more specific instructions for your setup!
