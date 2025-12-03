#!/bin/bash

# =============================================================================
# DEPLOYMENT SCRIPT FOR AGENTIC MULTIMODAL RAG SYSTEM
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="rag"
APP_NAME="agentic-multimodal-rag"
IMAGE_TAG="${1:-latest}"
REGISTRY="${REGISTRY:-registry.bionicaisolutions.com}"
IMAGE_NAME="${REGISTRY}/rag/${APP_NAME}:${IMAGE_TAG}"

echo -e "${BLUE}🚀 Deploying Agentic Multimodal RAG System${NC}"
echo -e "${BLUE}===========================================${NC}"
echo -e "Namespace: ${YELLOW}$NAMESPACE${NC}"
echo -e "Image: ${YELLOW}$IMAGE_NAME${NC}"
echo -e "Registry: ${YELLOW}$REGISTRY${NC}"
echo ""

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl is not installed or not in PATH${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ kubectl found${NC}"
}

# Function to check cluster connectivity
check_cluster() {
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Connected to Kubernetes cluster${NC}"
}

# Function to create namespace if it doesn't exist
create_namespace() {
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        echo -e "${GREEN}✅ Namespace $NAMESPACE already exists${NC}"
    else
        echo -e "${YELLOW}📦 Creating namespace $NAMESPACE${NC}"
        kubectl create namespace $NAMESPACE
        echo -e "${GREEN}✅ Namespace $NAMESPACE created${NC}"
    fi
}

# Function to apply Kubernetes manifests
apply_manifests() {
    echo -e "${YELLOW}📋 Applying Kubernetes manifests${NC}"
    
    # Apply manifests in order
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/rbac.yaml
    kubectl apply -f k8s/pvc.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secret.yaml
    
    # Update image in deployment
    sed "s|registry.bionicaisolutions.com/rag/agentic-multimodal-rag:.*|$IMAGE_NAME|g" k8s/deployment.yaml | kubectl apply -f -
    
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/ingress.yaml
    
    echo -e "${GREEN}✅ All manifests applied successfully${NC}"
}

# Function to wait for deployment
wait_for_deployment() {
    echo -e "${YELLOW}⏳ Waiting for deployment to be ready${NC}"
    kubectl rollout status deployment/rag-app -n $NAMESPACE --timeout=300s
    echo -e "${GREEN}✅ Deployment is ready${NC}"
}

# Function to show deployment status
show_status() {
    echo -e "${BLUE}📊 Deployment Status${NC}"
    echo -e "${BLUE}==================${NC}"
    
    echo -e "\n${YELLOW}Pods:${NC}"
    kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=$APP_NAME
    
    echo -e "\n${YELLOW}Services:${NC}"
    kubectl get services -n $NAMESPACE
    
    echo -e "\n${YELLOW}Ingress:${NC}"
    kubectl get ingress -n $NAMESPACE
    
    echo -e "\n${YELLOW}Deployment:${NC}"
    kubectl get deployment -n $NAMESPACE
}

# Function to run health check
health_check() {
    echo -e "${YELLOW}🏥 Running health check${NC}"
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get service rag-app-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    SERVICE_PORT=$(kubectl get service rag-app-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "80")
    
    if [ -z "$SERVICE_IP" ]; then
        echo -e "${YELLOW}⚠️  LoadBalancer IP not available, using port-forward${NC}"
        kubectl port-forward -n $NAMESPACE svc/rag-app-internal 8080:8000 &
        PORT_FORWARD_PID=$!
        sleep 5
        
        if curl -f http://localhost:8080/health &> /dev/null; then
            echo -e "${GREEN}✅ Health check passed${NC}"
            kill $PORT_FORWARD_PID 2>/dev/null || true
        else
            echo -e "${RED}❌ Health check failed${NC}"
            kill $PORT_FORWARD_PID 2>/dev/null || true
            exit 1
        fi
    else
        echo -e "${YELLOW}🌐 Testing external endpoint: http://$SERVICE_IP:$SERVICE_PORT/health${NC}"
        if curl -f http://$SERVICE_IP:$SERVICE_PORT/health &> /dev/null; then
            echo -e "${GREEN}✅ Health check passed${NC}"
        else
            echo -e "${RED}❌ Health check failed${NC}"
            exit 1
        fi
    fi
}

# Function to show logs
show_logs() {
    echo -e "${YELLOW}📋 Recent logs from deployment${NC}"
    kubectl logs -n $NAMESPACE -l app.kubernetes.io/name=$APP_NAME --tail=50
}

# Function to cleanup (for testing)
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up deployment${NC}"
    kubectl delete namespace $NAMESPACE --ignore-not-found=true
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Main deployment function
deploy() {
    echo -e "${BLUE}Starting deployment process...${NC}"
    
    check_kubectl
    check_cluster
    create_namespace
    apply_manifests
    wait_for_deployment
    show_status
    health_check
    
    echo -e "\n${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${BLUE}===========================================${NC}"
    
    # Show access information
    SERVICE_IP=$(kubectl get service rag-app-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    if [ -n "$SERVICE_IP" ]; then
        echo -e "${GREEN}🌐 Application is accessible at: http://$SERVICE_IP${NC}"
    else
        echo -e "${YELLOW}🔗 Use port-forward to access the application:${NC}"
        echo -e "${YELLOW}   kubectl port-forward -n $NAMESPACE svc/rag-app-internal 8080:8000${NC}"
        echo -e "${YELLOW}   Then visit: http://localhost:8080${NC}"
    fi
    
    echo -e "\n${BLUE}📚 Useful commands:${NC}"
    echo -e "${YELLOW}   View logs: kubectl logs -n $NAMESPACE -l app.kubernetes.io/name=$APP_NAME -f${NC}"
    echo -e "${YELLOW}   Scale deployment: kubectl scale -n $NAMESPACE deployment/rag-app --replicas=3${NC}"
    echo -e "${YELLOW}   Delete deployment: kubectl delete namespace $NAMESPACE${NC}"
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    "health")
        health_check
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|"-h"|"--help")
        echo -e "${BLUE}Usage: $0 [COMMAND] [IMAGE_TAG]${NC}"
        echo ""
        echo -e "${YELLOW}Commands:${NC}"
        echo -e "  deploy   Deploy the application (default)"
        echo -e "  status   Show deployment status"
        echo -e "  logs     Show application logs"
        echo -e "  health   Run health check"
        echo -e "  cleanup  Remove the deployment"
        echo -e "  help     Show this help message"
        echo ""
        echo -e "${YELLOW}Examples:${NC}"
        echo -e "  $0 deploy latest"
        echo -e "  $0 deploy v1.0.0"
        echo -e "  $0 status"
        echo -e "  $0 logs"
        ;;
    *)
        # If first argument is not a command, treat it as image tag
        deploy
        ;;
esac
