<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dynamo + Envoy AI Gateway

This guide explains how to use [Envoy AI Gateway](https://github.com/envoyproxy/ai-gateway) as the inference gateway for Nvidia Dynamo, replacing or complementing the default kGateway/HTTPRoute setup.

## What Envoy AI Gateway adds

| Capability | kGateway (HTTPRoute) | Envoy AI Gateway (AIGatewayRoute) |
|---|---|---|
| KV-aware routing (Dynamo EPP) | Yes | Yes |
| OpenAI schema normalization | No | Yes — model name extracted from body |
| Multi-model routing | Manual header match | Native `x-ai-eg-model` header routing |
| Token-level rate limiting | No | Yes (`QuotaPolicy`) |
| JWT / API-key auth | Manual EnvoyFilter | Native `BackendSecurityPolicy` |
| External provider fallback | No | Yes (route rule to `AIServiceBackend`) |
| Token usage metrics | No | Yes (Prometheus + OpenTelemetry) |

## Architecture

```
Client (OpenAI SDK)
        │  POST /v1/chat/completions
        │  -H "x-ai-eg-model: Qwen/Qwen3-0.6B"
        ▼
Envoy AI Gateway (AIGatewayRoute)
        │  matches x-ai-eg-model header
        │  extracts model name, counts tokens
        ▼ ext-proc gRPC (GAIE InferencePool EPP)
Dynamo EPP (port 9002)
        │  KV overlap scoring (radix tree, NATS events)
        │  sets x-worker-instance-id header
        ▼ x-gateway-destination-endpoint: <pod-ip>:8000
Dynamo Frontend sidecar pod
        │  --router-mode direct
        ▼
vLLM / SGLang engine
```

## Prerequisites

Install the required cluster components once per cluster:

```bash
chmod +x deploy/inference-gateway/envoy-ai-gateway/scripts/install-prerequisites.sh
./deploy/inference-gateway/envoy-ai-gateway/scripts/install-prerequisites.sh
```

This installs:
- Envoy Gateway v1.4.0
- Envoy AI Gateway v0.5.0
- Gateway API Inference Extension CRDs v1.2.1

> You also need a running Dynamo platform (NATS JetStream + operator) and a
> `DynamoGraphDeployment`. Follow the main [Kubernetes deployment guide](./deployment.md)
> first if you have not done so.

### Create a Gateway resource

The `AIGatewayRoute` attaches to a `Gateway`. Create one if it does not exist:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: inference-gateway
  namespace: my-model
spec:
  gatewayClassName: envoy-gateway
  listeners:
    - name: http
      protocol: HTTP
      port: 80
```

## Deploy

```bash
export EPP_IMAGE="nvcr.io/nvidia/ai-dynamo/epp-image:<tag>"

helm upgrade --install qwen \
  deploy/inference-gateway/envoy-ai-gateway/helm/dynamo-eagw \
  --namespace my-model \
  --create-namespace \
  --set-string extension.image=$EPP_IMAGE \
  -f deploy/inference-gateway/envoy-ai-gateway/examples/vllm_agg_qwen.yaml
```

### Disaggregated mode

```bash
helm upgrade --install qwen-disagg \
  deploy/inference-gateway/envoy-ai-gateway/helm/dynamo-eagw \
  --namespace my-model \
  --create-namespace \
  --set-string extension.image=$EPP_IMAGE \
  -f deploy/inference-gateway/envoy-ai-gateway/examples/vllm_disagg_qwen.yaml
```

## Verify

```bash
# Check EPP is ready
kubectl rollout status deployment/qwen-epp -n my-model

# Get the Gateway's external IP
kubectl get gateway inference-gateway -n my-model

# Send a test request
GATEWAY_IP=$(kubectl get svc -n envoy-gateway-system \
  -l gateway.envoyproxy.io/owning-gateway-name=inference-gateway \
  -o jsonpath='{.items[0].status.loadBalancer.ingress[0].ip}')

curl http://$GATEWAY_IP/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-ai-eg-model: Qwen/Qwen3-0.6B" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

## Configuration reference

All values are documented in
`deploy/inference-gateway/envoy-ai-gateway/helm/dynamo-eagw/values.yaml`.

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `dynamoGraphDeploymentName` | `vllm-agg` | Name of the `DynamoGraphDeployment` |
| `model.identifier` | `Qwen/Qwen3-0.6B` | Full model name matched in `x-ai-eg-model` |
| `model.shortName` | `qwen` | Short name for resource naming |
| `aiGatewayRoute.gatewayName` | `inference-gateway` | Gateway to attach the route to |
| `aiGatewayRoute.fallback.enabled` | `false` | Enable fallback rule to external provider |
| `envoyGateway.patchEnabled` | `true` | Apply `EnvoyGateway` config patch |
| `epp.useDynamo` | `true` | Enable KV-aware routing via Dynamo EPP |
| `epp.dynamo.useEtcd` | `false` | Use etcd instead of Kubernetes for discovery |
| `extension.image` | *(required)* | EPP container image |

## Token rate limiting

Add a `QuotaPolicy` to limit token consumption per client:

```yaml
apiVersion: aigateway.envoyproxy.io/v1alpha1
kind: QuotaPolicy
metadata:
  name: qwen-quota
  namespace: my-model
spec:
  targetRefs:
    - name: qwen-route
      kind: AIGatewayRoute
      group: aigateway.envoyproxy.io
  limits:
    - type: Token
      value: 100000
      unit: Minute
```

## External provider fallback

To fall back to OpenAI when a different model name is requested:

```yaml
# In values override:
aiGatewayRoute:
  fallback:
    enabled: true
    modelName: "gpt-4o-mini"
    backendName: "openai-backend"
```

Then create the `AIServiceBackend` and its `BackendSecurityPolicy` separately (see
[Envoy AI Gateway docs](https://github.com/envoyproxy/ai-gateway/tree/main/docs)).

## KV-aware routing (EPP) internals

The Dynamo EPP implements the same KV cache-aware routing algorithm as the
[`lib/kv-router`](../../lib/kv-router/) Rust library:

1. Workers publish cache block events to NATS JetStream.
2. The EPP (via CGO → `libdynamo_llm_capi.a`) maintains a per-worker radix tree
   of cached token block hashes.
3. For each request, the EPP tokenizes the prompt, computes overlap scores
   (`logit = overlap_weight × prefill_blocks + decode_blocks`), and selects the
   worker with the minimum logit (softmax-sampled at nonzero temperature).
4. The GAIE framework translates the selected worker ID to a pod IP and sets the
   `x-gateway-destination-endpoint` header for Envoy's original-destination routing.

See [`deploy/inference-gateway/epp/`](../deploy/inference-gateway/epp/) for the EPP
source code.

## Differences from the kGateway (GAIE) integration

The `dynamo-gaie` chart at
`deploy/inference-gateway/standalone/helm/dynamo-gaie/` uses `HTTPRoute` and
kGateway. The `dynamo-eagw` chart uses `AIGatewayRoute` and Envoy AI Gateway.

Both charts use the **same EPP image** and the **same `InferencePool` CRD**.
You can run both in the same cluster on different Gateway resources.

## Troubleshooting

**EPP pod stuck in `Init` state**
- Check NATS connectivity: `kubectl exec -n my-model <epp-pod> -- nc -zv <nats-host> 4222`
- Verify `DynamoWorkerMetadata` CRDs exist: `kubectl get dynamoworkermetadatas -A`

**Requests returning 503**
- Ensure `InferencePool` shows ready endpoints:
  `kubectl describe inferencepool qwen-pool -n my-model`
- Check the EPP logs: `kubectl logs -n my-model -l app=qwen-epp`

**EnvoyGateway patch conflict**
- If you manage `EnvoyGateway` via another tool, set `envoyGateway.patchEnabled: false`
  and apply the equivalent config manually (see
  `templates/envoy-gateway-config.yaml` for the required fields).
