#!/bin/bash
# Deployment script for Agentic Multimodal RAG System
# Run this script with appropriate Kubernetes permissions (cluster-admin or namespace admin)

set -e

NAMESPACE="rag"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Deploying Agentic Multimodal RAG System"
echo "Namespace: $NAMESPACE"
echo "=========================================="
echo ""

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo "Creating namespace: $NAMESPACE"
    kubectl apply -f "$SCRIPT_DIR/namespace.yaml"
    echo "✓ Namespace created"
else
    echo "✓ Namespace '$NAMESPACE' already exists"
fi

echo ""
echo "Step 1/7: Applying RBAC (ServiceAccount, Role, RoleBinding)..."
kubectl apply -f "$SCRIPT_DIR/rbac.yaml"
echo "✓ RBAC applied"

echo ""
echo "Step 2/7: Creating PersistentVolumeClaims..."
kubectl apply -f "$SCRIPT_DIR/pvc.yaml"
echo "✓ PVCs created"

echo ""
echo "Step 3/7: Applying Secrets..."
kubectl apply -f "$SCRIPT_DIR/secret.yaml"
echo "✓ Secrets applied"

echo ""
echo "Step 4/7: Applying ConfigMap..."
kubectl apply -f "$SCRIPT_DIR/configmap.yaml"
echo "✓ ConfigMap applied"

echo ""
echo "Step 5/7: Creating Services..."
kubectl apply -f "$SCRIPT_DIR/service.yaml"
echo "✓ Services created"

echo ""
echo "Step 6/7: Deploying Application..."
kubectl apply -f "$SCRIPT_DIR/deployment.yaml"
echo "✓ Deployment created"

echo ""
echo "Step 7/7: Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/rag-app -n "$NAMESPACE" || true

echo ""
echo "=========================================="
echo "Deployment Summary"
echo "=========================================="
echo ""
echo "Pods:"
kubectl get pods -n "$NAMESPACE"
echo ""
echo "Services:"
kubectl get svc -n "$NAMESPACE"
echo ""
echo "Deployment Status:"
kubectl get deployment -n "$NAMESPACE"
echo ""
echo "To check pod logs:"
echo "  kubectl logs -n $NAMESPACE -l app.kubernetes.io/name=agentic-multimodal-rag -f"
echo ""
echo "To get service endpoints:"
echo "  kubectl get svc -n $NAMESPACE"
echo ""
echo "Deployment complete!"

