// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Envoy ext_proc service that implements KV-cache-aware routing for Dynamo.
//!
//! # Architecture
//!
//! Envoy opens one bidirectional gRPC stream per HTTP request (via the
//! `ExternalProcessor` service).  Each stream goes through up to four phases:
//!
//! ```text
//! Envoy  ──request_headers──►  ExtProc  (inspect headers, opt. tell Envoy to buffer body)
//! Envoy  ──request_body─────►  ExtProc  (tokenise prompt → find_matches → set destination)
//! Envoy  ──response_headers─►  ExtProc  (pass through; on first chunk → mark prefill done)
//! Envoy  ──response_body────►  ExtProc  (pass through; stream close → free request state)
//! ```
//!
//! The routing decision is communicated to Envoy by setting
//! `x-gateway-destination-endpoint: <pod-ip>:<port>` in the `CommonResponse`
//! header mutation returned during the `request_body` phase.
//!
//! # Integration with KVRouter
//!
//! The service wraps a [`dynamo_kv_router::indexer::KvIndexerInterface`] impl
//! (typically a [`dynamo_kv_router::ThreadPoolIndexer`] over
//! [`dynamo_kv_router::ConcurrentRadixTreeCompressed`]).  Workers publish their
//! KV cache events to the indexer via the [`WorkerMap`] registration API and
//! NATS/ZMQ transport (external to this crate).

pub mod config;
pub mod routing;
pub mod service;
pub mod worker_map;

/// Protobuf types generated from `proto/envoy_ext_proc.proto`.
pub mod proto {
    tonic::include_proto!("envoy.service.ext_proc.v3");
}

#[cfg(test)]
mod tests;
