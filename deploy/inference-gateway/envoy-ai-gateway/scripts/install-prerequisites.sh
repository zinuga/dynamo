#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install prerequisites for Dynamo + Envoy AI Gateway.
# Run this once per cluster before deploying the dynamo-eagw Helm chart.
#
# Usage:
#   chmod +x install-prerequisites.sh
#   ./install-prerequisites.sh

set -euo pipefail

ENVOY_GATEWAY_VERSION="${ENVOY_GATEWAY_VERSION:-v1.4.0}"
ENVOY_AI_GATEWAY_VERSION="${ENVOY_AI_GATEWAY_VERSION:-v0.5.0}"
GAIE_VERSION="${GAIE_VERSION:-v1.2.1}"

echo "==> Installing Gateway API CRDs"
kubectl apply -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${ENVOY_GATEWAY_VERSION}/experimental-install.yaml"

echo "==> Installing Envoy Gateway ${ENVOY_GATEWAY_VERSION}"
helm upgrade --install envoy-gateway \
  oci://docker.io/envoyproxy/gateway-helm \
  --version "${ENVOY_GATEWAY_VERSION}" \
  --namespace envoy-gateway-system \
  --create-namespace \
  --wait

echo "==> Installing Envoy AI Gateway ${ENVOY_AI_GATEWAY_VERSION}"
helm upgrade --install ai-gateway \
  oci://docker.io/envoyproxy/ai-gateway-helm \
  --version "${ENVOY_AI_GATEWAY_VERSION}" \
  --namespace envoy-ai-gateway-system \
  --create-namespace \
  --set envoyGateway.enabled=false \
  --wait

echo "==> Installing Gateway API Inference Extension CRDs ${GAIE_VERSION}"
kubectl apply -f \
  "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${GAIE_VERSION}/manifests.yaml"

echo ""
echo "Prerequisites installed successfully."
echo "Next: deploy the dynamo-eagw Helm chart."
