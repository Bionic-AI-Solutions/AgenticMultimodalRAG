#!/bin/bash

# =============================================================================
# BUILD AND PUSH DOCKER IMAGE TO REGISTRY
# =============================================================================
# This script builds the Docker image and pushes it to registry.bionicaisolutions.com

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGISTRY="registry.bionicaisolutions.com"
IMAGE_NAME="rag/agentic-multimodal-rag"
IMAGE_TAG="${1:-latest}"
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${BLUE}🐳 Building and Pushing Docker Image${NC}"
echo -e "${BLUE}=====================================${NC}"
echo -e "Registry: ${YELLOW}$REGISTRY${NC}"
echo -e "Image: ${YELLOW}$IMAGE_NAME${NC}"
echo -e "Tag: ${YELLOW}$IMAGE_TAG${NC}"
echo -e "Full Image: ${YELLOW}$FULL_IMAGE_NAME${NC}"
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed or not in PATH${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker found${NC}"
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        echo -e "${RED}❌ Docker daemon is not running${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker daemon is running${NC}"
}

# Function to login to registry
login_to_registry() {
    echo -e "${YELLOW}🔐 Logging into registry...${NC}"
    
    if [ -z "$REGISTRY_USERNAME" ] || [ -z "$REGISTRY_PASSWORD" ]; then
        echo -e "${YELLOW}⚠️  REGISTRY_USERNAME and REGISTRY_PASSWORD not set${NC}"
        echo -e "${YELLOW}💡 Attempting to login interactively...${NC}"
        docker login "$REGISTRY"
    else
        echo "$REGISTRY_PASSWORD" | docker login "$REGISTRY" -u "$REGISTRY_USERNAME" --password-stdin
    fi
    
    echo -e "${GREEN}✅ Logged into registry${NC}"
}

# Function to build Docker image
build_image() {
    echo -e "${YELLOW}🔨 Building Docker image...${NC}"
    
    # Get build arguments
    BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    VERSION="${IMAGE_TAG}"
    
    docker build \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$VCS_REF" \
        --build-arg VERSION="$VERSION" \
        -t "$FULL_IMAGE_NAME" \
        -t "${REGISTRY}/${IMAGE_NAME}:latest" \
        .
    
    echo -e "${GREEN}✅ Image built successfully${NC}"
}

# Function to push Docker image
push_image() {
    echo -e "${YELLOW}📤 Pushing Docker image to registry...${NC}"
    
    docker push "$FULL_IMAGE_NAME"
    
    # Also push latest tag if not already latest
    if [ "$IMAGE_TAG" != "latest" ]; then
        docker push "${REGISTRY}/${IMAGE_NAME}:latest"
    fi
    
    echo -e "${GREEN}✅ Image pushed successfully${NC}"
}

# Function to show image info
show_image_info() {
    echo ""
    echo -e "${GREEN}🎉 Build and push completed successfully!${NC}"
    echo -e "${BLUE}===========================================${NC}"
    echo -e "${YELLOW}Image Details:${NC}"
    echo -e "  Registry: ${BLUE}$REGISTRY${NC}"
    echo -e "  Image: ${BLUE}$IMAGE_NAME${NC}"
    echo -e "  Tag: ${BLUE}$IMAGE_TAG${NC}"
    echo -e "  Full Name: ${BLUE}$FULL_IMAGE_NAME${NC}"
    echo ""
    echo -e "${YELLOW}📋 Next Steps:${NC}"
    echo -e "  1. Verify image in registry:"
    echo -e "     docker pull $FULL_IMAGE_NAME"
    echo -e "  2. Update Kubernetes deployment:"
    echo -e "     kubectl set image deployment/rag-app rag-app=$FULL_IMAGE_NAME -n rag"
    echo -e "  3. Or use the deployment script:"
    echo -e "     ./scripts/deploy.sh deploy $IMAGE_TAG"
    echo ""
}

# Main function
main() {
    check_prerequisites
    login_to_registry
    build_image
    push_image
    show_image_info
}

# Parse command line arguments
case "${1:-build}" in
    "build"|"")
        main
        ;;
    "help"|"-h"|"--help")
        echo -e "${BLUE}Usage: $0 [IMAGE_TAG]${NC}"
        echo ""
        echo -e "${YELLOW}Arguments:${NC}"
        echo -e "  IMAGE_TAG    Docker image tag (default: latest)"
        echo ""
        echo -e "${YELLOW}Environment Variables:${NC}"
        echo -e "  REGISTRY_USERNAME    Registry username (optional, will prompt if not set)"
        echo -e "  REGISTRY_PASSWORD    Registry password (optional, will prompt if not set)"
        echo ""
        echo -e "${YELLOW}Examples:${NC}"
        echo -e "  $0 latest"
        echo -e "  $0 v1.0.0"
        echo -e "  $0 facade-v1"
        echo -e "  REGISTRY_USERNAME=user REGISTRY_PASSWORD=pass $0 latest"
        ;;
    *)
        # Treat argument as image tag
        IMAGE_TAG="$1"
        main
        ;;
esac

