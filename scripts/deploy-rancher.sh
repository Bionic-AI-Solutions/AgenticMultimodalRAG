#!/bin/bash

# =============================================================================
# RANCHER DEPLOYMENT SCRIPT FOR AGENTIC MULTIMODAL RAG SYSTEM
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
RANCHER_URL="${RANCHER_URL:-}"
RANCHER_TOKEN="${RANCHER_TOKEN:-}"

echo -e "${BLUE}🚀 Rancher Deployment for Agentic Multimodal RAG System${NC}"
echo -e "${BLUE}=====================================================${NC}"
echo -e "Namespace: ${YELLOW}$NAMESPACE${NC}"
echo -e "Image: ${YELLOW}$IMAGE_NAME${NC}"
echo -e "Registry: ${YELLOW}$REGISTRY${NC}"
echo -e "Rancher URL: ${YELLOW}${RANCHER_URL:-'Not set'}"${NC}
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl is not installed or not in PATH${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ kubectl found${NC}"
    
    # Check if Rancher CLI is available (optional)
    if command -v rancher &> /dev/null; then
        echo -e "${GREEN}✅ Rancher CLI found${NC}"
        RANCHER_CLI_AVAILABLE=true
    else
        echo -e "${YELLOW}⚠️  Rancher CLI not found (optional)${NC}"
        RANCHER_CLI_AVAILABLE=false
    fi
    
    # Check if we can connect to cluster
    if kubectl cluster-info &> /dev/null; then
        echo -e "${GREEN}✅ Connected to Kubernetes cluster${NC}"
    else
        echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
        echo -e "${YELLOW}💡 Make sure your kubeconfig is properly configured${NC}"
        exit 1
    fi
}

# Function to prepare manifests
prepare_manifests() {
    echo -e "${YELLOW}📋 Preparing Kubernetes manifests...${NC}"
    
    # Create temporary directory for modified manifests
    TEMP_DIR=$(mktemp -d)
    echo -e "${BLUE}📁 Using temporary directory: $TEMP_DIR${NC}"
    
    # Copy manifests to temp directory
    cp k8s/*.yaml "$TEMP_DIR/"
    
    # Update image in deployment
    sed -i.bak "s|registry.bionicaisolutions.com/rag/agentic-multimodal-rag:.*|$IMAGE_NAME|g" "$TEMP_DIR/deployment.yaml"
    
    echo -e "${GREEN}✅ Manifests prepared${NC}"
    echo "$TEMP_DIR"
}

# Function to deploy via kubectl
deploy_kubectl() {
    local temp_dir="$1"
    
    echo -e "${YELLOW}🚀 Deploying via kubectl...${NC}"
    
    # Apply manifests in order
    kubectl apply -f "$temp_dir/namespace.yaml"
    kubectl apply -f "$temp_dir/rbac.yaml"
    kubectl apply -f "$temp_dir/pvc.yaml"
    kubectl apply -f "$temp_dir/configmap.yaml"
    kubectl apply -f "$temp_dir/secret.yaml"
    kubectl apply -f "$temp_dir/deployment.yaml"
    kubectl apply -f "$temp_dir/service.yaml"
    kubectl apply -f "$temp_dir/ingress.yaml"
    
    echo -e "${GREEN}✅ All manifests applied successfully${NC}"
}

# Function to deploy via Rancher CLI
deploy_rancher_cli() {
    local temp_dir="$1"
    
    if [ "$RANCHER_CLI_AVAILABLE" = false ]; then
        echo -e "${YELLOW}⚠️  Rancher CLI not available, using kubectl instead${NC}"
        deploy_kubectl "$temp_dir"
        return
    fi
    
    echo -e "${YELLOW}🚀 Deploying via Rancher CLI...${NC}"
    
    # Login to Rancher if credentials provided
    if [ -n "$RANCHER_URL" ] && [ -n "$RANCHER_TOKEN" ]; then
        echo -e "${BLUE}🔐 Logging into Rancher...${NC}"
        rancher login "$RANCHER_URL" --token "$RANCHER_TOKEN"
    fi
    
    # Apply manifests using Rancher CLI
    rancher kubectl apply -f "$temp_dir/namespace.yaml"
    rancher kubectl apply -f "$temp_dir/rbac.yaml"
    rancher kubectl apply -f "$temp_dir/pvc.yaml"
    rancher kubectl apply -f "$temp_dir/configmap.yaml"
    rancher kubectl apply -f "$temp_dir/secret.yaml"
    rancher kubectl apply -f "$temp_dir/deployment.yaml"
    rancher kubectl apply -f "$temp_dir/service.yaml"
    rancher kubectl apply -f "$temp_dir/ingress.yaml"
    
    echo -e "${GREEN}✅ All manifests applied via Rancher CLI${NC}"
}

# Function to wait for deployment
wait_for_deployment() {
    echo -e "${YELLOW}⏳ Waiting for deployment to be ready...${NC}"
    
    # Wait for deployment to be ready
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
    echo -e "${YELLOW}🏥 Running health check...${NC}"
    
    # Port forward to test health endpoint
    echo -e "${BLUE}🔗 Setting up port forward for health check...${NC}"
    kubectl port-forward -n $NAMESPACE svc/rag-app-internal 8080:8000 &
    PORT_FORWARD_PID=$!
    
    # Wait for port forward to be ready
    sleep 10
    
    # Test health endpoint
    if curl -f http://localhost:8080/health &> /dev/null; then
        echo -e "${GREEN}✅ Health check passed${NC}"
    else
        echo -e "${RED}❌ Health check failed${NC}"
        kill $PORT_FORWARD_PID 2>/dev/null || true
        exit 1
    fi
    
    # Clean up port forward
    kill $PORT_FORWARD_PID 2>/dev/null || true
}

# Function to show access information
show_access_info() {
    echo -e "\n${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${BLUE}===========================================${NC}"
    
    # Get service information
    SERVICE_IP=$(kubectl get service rag-app-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [ -n "$SERVICE_IP" ]; then
        echo -e "${GREEN}🌐 Application is accessible at: http://$SERVICE_IP${NC}"
    else
        echo -e "${YELLOW}🔗 Access the application using port-forward:${NC}"
        echo -e "${YELLOW}   kubectl port-forward -n $NAMESPACE svc/rag-app-internal 8080:8000${NC}"
        echo -e "${YELLOW}   Then visit: http://localhost:8080${NC}"
    fi
    
    echo -e "\n${BLUE}📚 Useful commands:${NC}"
    echo -e "${YELLOW}   View logs: kubectl logs -n $NAMESPACE -l app.kubernetes.io/name=$APP_NAME -f${NC}"
    echo -e "${YELLOW}   Scale deployment: kubectl scale -n $NAMESPACE deployment/rag-app --replicas=3${NC}"
    echo -e "${YELLOW}   Delete deployment: kubectl delete namespace $NAMESPACE${NC}"
    
    echo -e "\n${BLUE}🌐 Rancher UI Access:${NC}"
    if [ -n "$RANCHER_URL" ]; then
        echo -e "${YELLOW}   Rancher UI: $RANCHER_URL${NC}"
        echo -e "${YELLOW}   Navigate to: Workloads → rag-system namespace${NC}"
    else
        echo -e "${YELLOW}   Access your Rancher UI to manage the deployment${NC}"
    fi
}

# Function to cleanup temporary files
cleanup() {
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
        echo -e "${GREEN}✅ Cleaned up temporary files${NC}"
    fi
}

# Function to show help
show_help() {
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
    echo -e "${YELLOW}Environment Variables:${NC}"
    echo -e "  REGISTRY        Registry URL (default: registry.bionicaisolutions.com)"
    echo -e "  RANCHER_URL     Rancher server URL (optional)"
    echo -e "  RANCHER_TOKEN   Rancher API token (optional)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  $0 deploy latest"
    echo -e "  $0 deploy v1.0.0"
    echo -e "  REGISTRY=registry.bionicaisolutions.com $0 deploy latest"
    echo -e "  RANCHER_URL=https://rancher.example.com $0 deploy"
    echo ""
}

# Main deployment function
deploy() {
    echo -e "${BLUE}Starting Rancher deployment process...${NC}"
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    check_prerequisites
    TEMP_DIR=$(prepare_manifests)
    
    # Choose deployment method
    if [ "$RANCHER_CLI_AVAILABLE" = true ] && [ -n "$RANCHER_URL" ]; then
        deploy_rancher_cli "$TEMP_DIR"
    else
        deploy_kubectl "$TEMP_DIR"
    fi
    
    wait_for_deployment
    show_status
    health_check
    show_access_info
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
        kubectl logs -n $NAMESPACE -l app.kubernetes.io/name=$APP_NAME -f
        ;;
    "health")
        health_check
        ;;
    "cleanup")
        kubectl delete namespace $NAMESPACE --ignore-not-found=true
        echo -e "${GREEN}✅ Cleanup completed${NC}"
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        # If first argument is not a command, treat it as image tag
        deploy
        ;;
esac
